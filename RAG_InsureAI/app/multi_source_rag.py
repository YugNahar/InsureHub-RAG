"""
Unified RAG with strict grounding – no hallucinations.
Supports document filtering with substring matching.
"""
import asyncio
import logging
import os
import re
import time
from typing import List, Tuple, Optional

try:
    from openai import APIConnectionError as _APIConnectionError, APITimeoutError as _APITimeoutError, APIStatusError as _APIStatusError
except ImportError:
    _APIConnectionError = Exception
    _APITimeoutError = Exception
    _APIStatusError = Exception

from rapidfuzz import fuzz, process
from turbovec_store import _rerank_windows, _get_shared_reranker
from metadata_tagger import (
    classify_query_policy_type, get_active_vocab, _valid_policy_types, _normalize_policy_type,
)
import contamination_trace

logger = logging.getLogger(__name__)

_LLM_BACKEND_ERRORS = (_APIConnectionError, _APITimeoutError, _APIStatusError)

# ── Typo-tolerant insurance vocabulary ──────────────────────────────────────────
# Correctly-spelled insurance-domain terms used by _correct_typos() to
# fix common typos before vector retrieval.
_INSURANCE_VOCAB = [
    "insurance", "policy", "premium", "deductible", "coverage", "claim", "claims",
    "insured", "insurer", "insurers", "underwriting", "underwrite", "renewal", "nominee",
    "beneficiary", "cashless", "reimbursement", "rider", "liability", "copay",
    "endorsement", "subrogation", "exclusion", "maturity", "surrender",
    "vehicle", "comprehensive", "third party", "cover", "covered", "co-pay",
    "deductibles", "premiums", "health", "medical", "hospital", "surgery",
    "prescription", "medication", "accident", "disability", "critical illness",
    "maternity", "dental", "vision", "agent", "broker", "benefit", "benefits",
    "term life", "whole life", "endowment", "annuity", "pension", "retirement",
    "investment", "savings", "finance", "financial", "limit", "limits",
    "no-claim bonus", "sum insured", "sum assured", "grace period",
    "waiting period", "free look period", "cancellation", "protection",
    "calculation", "formula", "amount", "documents", "process", "settlement",
]

# Ordinary English words that fuzzy-match an _INSURANCE_VOCAB entry above
# the 85 threshold but are never actually a typo of it — raising the
# threshold from 80 to 85 (see _correct_typos) closed off the "detail" ->
# "dental" class of false positive, but a systematic scan of the top
# 10,000 most common English words turned up 19 more words that still
# score >=85, in the same range as genuine typos ("nominie" -> "nominee"
# also scores 85.7), so no threshold value can separate them from real
# typo-correction coverage. These are hand-verified as real, common
# words that would never legitimately need fixing to the paired vocab
# term, and are excluded outright before the fuzzy match ever runs.
# Worst cases if left unprotected: "injured" -> "insured" (opposite-ish
# meaning, and "injured" is one of the most common words in accident/
# health queries), "requirement" -> "retirement" (silently redirects an
# eligibility question into unrelated retirement-planning content),
# "description" -> "prescription", "copy" -> "copay". A few borderline
# scan hits ("heath" -> "health", "accent" -> "accident", "cove" ->
# "cover") were deliberately left OUT of this set: in an insurance-chat
# context a typo is far more likely than the literal rare word, so
# correction is still the right default for those three.
_TYPO_CORRECTION_PROTECTED_WORDS = frozenset({
    "injured", "requirement", "requirements", "description", "descriptions",
    "copy", "copies", "converted", "convert", "projection", "projections",
    "beneficial", "renewable", "document", "ride", "rides", "saving",
    "broke", "protecting", "tension", "meditation", "calculator",
})


# ── Casual greetings (fuzzy-matched) ─────────────────────────────────────────────
_CASUAL_GREETINGS = frozenset({
    "hi", "hello", "hey", "yo", "sup", "what's up", "whats up",
    "good morning", "good evening", "good afternoon",
    "howdy", "hiya", "dude", "bro", "hii", "heya",
})


# Common prefixes that turn a vocab word into a distinct compound insurance
# term (re+insurance, co+insurance, un+insured, under+insured, over+insurance,
# non+insurance). Users often type these with a space or hyphen instead of
# joined — "re insurance" / "re-insurance" — which changes nothing for a
# human reader but is a different token for retrieval than the corpus's
# actual joined spelling. _join_split_compounds() below re-joins them.
_COMPOUND_PREFIXES = ("re", "co", "un", "under", "over", "non")
_COMPOUND_PREFIX_RE = re.compile(
    r'\b(' + '|'.join(_COMPOUND_PREFIXES) + r')[\s-]+(\w{4,})\b',
    re.IGNORECASE,
)

# The only roots that actually form a real compound word with one of the
# prefixes above. Deliberately NOT the full _INSURANCE_VOCAB — that was
# the bug: matching against all 40+ vocab entries meant ANY of them
# following "under"/"re"/"co"/etc. got silently glued together, including
# words that don't form a real compound at all. Confirmed live: "covered
# under health insurance" got rewritten to "covered underhealth
# insurance" — "health" is a legitimate vocab entry on its own, but
# "underhealth" isn't a word, and the garbled query made retrieval
# unreliable (intermittent reranker-gate refusals on a question the KB
# answers fine). Same failure shape as "reinsurance" -> "insurance" in
# _correct_typos below: a broad match target catching real domain words
# that happen to sit near a prefix, not just genuine split compounds.
_COMPOUND_ROOTS = ["insurance", "insured", "insurer", "writing", "write"]


def _join_split_compounds(text: str) -> str:
    """Re-join a compound prefix ("re", "co", "un", "under", "over", "non")
    that was typed with a space or hyphen before a word matching a
    _COMPOUND_ROOTS entry — "re insurance" / "re-insurance" -> "reinsurance",
    "under insured" -> "underinsured" — so retrieval sees the same single
    token the knowledge base actually uses, instead of two separate ones.

    Only joins when the word after the prefix fuzzy-matches one of the
    small set of roots that actually form a real compound (score >= 85 —
    see _correct_typos for why this was raised from 80) — NOT the full
    insurance vocabulary, which would match ordinary domain words that
    happen to follow a prefix without forming any real compound term
    ("under health" is not "underhealth"). An unrelated word right after
    "re"/"co"/etc. ("re apply") is left untouched either way.
    """
    def _try_join(m: re.Match) -> str:
        prefix, rest = m.group(1), m.group(2)
        if rest.lower() in _TYPO_CORRECTION_PROTECTED_WORDS:
            return m.group(0)
        result = process.extractOne(rest, _COMPOUND_ROOTS, scorer=fuzz.ratio)
        if result is not None:
            best_match, score, _ = result
            if score >= 85:
                return f"{prefix.lower()}{best_match}"
        return m.group(0)

    return _COMPOUND_PREFIX_RE.sub(_try_join, text)


def _correct_typos(text: str) -> str:
    """Fix common typos in insurance-domain terms using fuzzy matching.

    Splits *text* into words. For each word of length >= 4, finds the best
    match in ``_INSURANCE_VOCAB`` using ``rapidfuzz`` with ``fuzz.ratio``.
    If the best match score >= 85 and is not already exact, replaces the
    word with the correctly-spelled vocabulary term. Returns the corrected
    sentence with original word order preserved.

    Skips the correction when the matched vocab word is fully contained
    inside a *longer* candidate word (e.g. "insurance" inside
    "reinsurance") — that's a legitimate compound/prefixed domain term
    built on top of the vocab word, not a misspelling of it. A real typo
    is a substitution/transposition/omission that stays roughly the same
    length as the intended word ("deductable" -> "deductible"); it doesn't
    cleanly embed a complete shorter word plus extra characters. Without
    this guard, "reinsurance" scores 90 against "insurance" (they share
    it as a substring) and gets silently rewritten to "insurance" before
    retrieval ever runs — and the same happens to "coinsurance",
    "uninsured", "underinsured", etc. against "insurance"/"insured". This
    check generalizes the fix so every such compound survives untouched
    without having to hand-list each one in the vocab.

    Also skips the reverse shape: a complete, shorter candidate word that
    happens to be the tail end of a longer vocab word with one short
    prefix chopped off — e.g. "over" scores 89 against "cover" (drop the
    leading "c" and they're identical) and would otherwise get "corrected"
    into "cover" any time it appears in an ordinary sentence ("what happens
    over the policy term"). Real typos almost never drop exactly the
    leading character(s) of a word; that's specifically the shape of an
    unrelated, independently-valid short word.

    Threshold raised 80 -> 85 after finding this was silently corrupting
    ordinary English words that happen to be near-neighbors of a vocab
    term, with no unusual-shape guard able to catch it since these are
    same-length substitutions, structurally identical to a real typo.
    Confirmed live: "explain fire insurance in detail" retrieved as
    "...in dental" (score 83.3 against "dental") and returned a flat "not
    in my knowledge base" refusal for a topic the KB actually covers well
    — silently corrupted before retrieval ever ran, so nothing downstream
    had a chance to catch it. A vocabulary scan turned up "company" ->
    "copay" (83.3) and "order" -> "rider" (80.0) as the same class of
    false positive. Every genuine typo this function is meant to catch
    ("deductable", "insurence", "premiu", "materinty", "hospitl", ...)
    scored 85.7 or higher in the same scan, so 85 cleanly separates real
    typos from coincidental collisions with common English words without
    losing any tested correction case.
    """
    text = _join_split_compounds(text)
    words = text.split()
    corrected = []
    for w in words:
        stripped = w.strip(".,!?;:()[]{}'\"")
        if len(stripped) >= 4 and stripped.lower() not in _TYPO_CORRECTION_PROTECTED_WORDS:
            result = process.extractOne(stripped, _INSURANCE_VOCAB, scorer=fuzz.ratio)
            if result is not None:
                best_match, score, _ = result
                stripped_lower = stripped.lower()
                best_match_lower = best_match.lower()
                is_compound_extension = (
                    len(stripped) > len(best_match)
                    and best_match_lower in stripped_lower
                )
                is_truncated_vocab_word = (
                    len(best_match) > len(stripped)
                    and best_match_lower.endswith(stripped_lower)
                )
                if (score >= 85 and best_match != stripped
                        and not is_compound_extension and not is_truncated_vocab_word):
                    prefix = w[:len(w) - len(w.lstrip(".,!?;:()[]{}'\""))]
                    suffix = w[len(w.rstrip(".,!?;:()[]{}'\"") or len(w)):]
                    corrected.append(f"{prefix}{best_match}{suffix}")
                    continue
        corrected.append(w)
    return " ".join(corrected)


# Bidirectional abbreviation <-> full-term expansion, appended to (never
# replacing) the retrieval query — confirmed live (2026-07-13): a KB
# document titled "Major Insurance Brokers in the GCC" ranked BELOW an
# unrelated, generic PDF when the user asked about "the Gulf Cooperation
# Council" instead of "GCC". The document's own text only ever uses the
# abbreviation, so a query using the spelled-out form has no lexical
# match against it at all — embedding similarity alone wasn't enough to
# keep it top-ranked against exact-term competition elsewhere in the KB.
# Appending the counterpart form (in whichever direction the query used)
# gives retrieval a lexical match against source documents that only
# ever use one form or the other. Appending rather than substituting
# keeps the original query intact for every other check downstream
# (lexical coverage, grounding, KV cache key) — this only adds extra
# surface area for the vector/hybrid search to match against.
_ABBREVIATION_EXPANSIONS = {
    "gcc": "gulf cooperation council",
    "uae": "united arab emirates",
    "ksa": "kingdom of saudi arabia",
    "irdai": "insurance regulatory and development authority of india",
}
_ABBREVIATION_RE = re.compile(
    r'\b(' + '|'.join(re.escape(k) for k in _ABBREVIATION_EXPANSIONS) + r')\b',
    re.IGNORECASE,
)
_EXPANSION_TO_ABBR = {v: k.upper() for k, v in _ABBREVIATION_EXPANSIONS.items()}
_EXPANSION_RE = re.compile(
    r'\b(' + '|'.join(re.escape(v) for v in _ABBREVIATION_EXPANSIONS.values()) + r')\b',
    re.IGNORECASE,
)


def _expand_abbreviations(text: str) -> str:
    """Append the counterpart form (abbreviation <-> spelled-out term) for
    any known regional/regulatory term found in *text*, so retrieval
    matches KB documents using either form. New abbreviations can be added
    to _ABBREVIATION_EXPANSIONS as they're found to matter for documents
    actually in the KB — this isn't meant to be an exhaustive dictionary,
    just a targeted fix for the specific class of mismatch confirmed live.
    """
    additions = []
    for m in _ABBREVIATION_RE.finditer(text):
        full = _ABBREVIATION_EXPANSIONS[m.group(1).lower()]
        if full not in additions:
            additions.append(full)
    for m in _EXPANSION_RE.finditer(text):
        abbr = _EXPANSION_TO_ABBR[m.group(1).lower()]
        if abbr not in additions:
            additions.append(abbr)
    if not additions:
        return text
    return text + " " + " ".join(additions)


# ── Context coverage check ────────────────────────────────────────────────────
# Domain-generic terms that appear in virtually every insurance chunk and
# therefore give no signal about whether a chunk covers the specific query topic.
_QUERY_STOP_WORDS = {
    # English function words
    'what', 'is', 'are', 'the', 'a', 'an', 'in', 'of', 'for', 'how', 'does',
    'do', 'i', 'my', 'me', 'by', 'with', 'under', 'about', 'can', 'will',
    'which', 'when', 'where', 'to', 'and', 'or', 'at', 'this', 'that', 'it',
    'its', 'be', 'been', 'has', 'have', 'had', 'any', 'all', 'from', 'on',
    # Conversational intent words — appear in questions but never in KB chunks
    'tell', 'you', 'your', 'give', 'explain', 'know', 'say', 'get', 'let',
    'show', 'help', 'talk', 'discuss', 'find', 'see', 'our', 'their', 'use',
    'used', 'using', 'work', 'works', 'mean', 'means', 'please', 'want',
    'need', 'like', 'just', 'also', 'more', 'some', 'very', 'well', 'good',
    'different', 'types', 'type', 'kind', 'kinds', 'various', 'general',
    'basic', 'main', 'key', 'between', 'difference', 'example', 'examples',
    # Answer-style/format requests — ask HOW to phrase the answer, never
    # appear as literal words inside actual KB content (no policy document
    # says "explain in simple words"). Reformulated follow-ups like "explain
    # it in simple words" or "give me an example" fold these into the
    # retrieval query, and requiring them to co-occur with the real topic
    # in a source chunk always fails regardless of whether the topic itself
    # is genuinely covered.
    'simple', 'simply', 'definition', 'define', 'detail', 'detailed',
    'briefly', 'brief', 'short', 'layman', 'easy', 'explanation',
    'language', 'words', 'scenario', 'illustration', 'instance',
    # Generic quality/selection words — appear in virtually every insurance doc
    # so they cannot indicate topic coverage (e.g. "best coverage", "choose a plan")
    'best', 'top', 'great', 'better', 'right', 'ideal', 'suitable', 'suited',
    'choose', 'choosing', 'select', 'selecting', 'pick', 'decide', 'deciding',
    'compare', 'comparing', 'recommend', 'recommended', 'important', 'affordable',
    'option', 'options', 'available', 'offer', 'offers', 'provide', 'provides',
    # Domain-generic insurance terms that appear in almost every chunk
    'insurance', 'insured', 'insurer', 'policy', 'policies', 'cover', 'coverage',
    'plan', 'claim', 'claims', 'benefits', 'benefit',
    # Generic time/measurement words — too common to indicate topic coverage
    'period', 'time', 'duration', 'date', 'days', 'year', 'month', 'months',
}

def _extract_topic_terms(query: str) -> set[str]:
    """Return discriminating topic words from a query after stop-word filtering."""
    stop = _QUERY_STOP_WORDS
    return {
        w for w in re.findall(r'\b[a-z]{3,}\b', query.lower())
        if not _is_stopword(w, stop)
    }


def _word_matches(topic: str, chunk_word: str) -> bool:
    """True if topic and chunk_word are the same root.

    Uses a 5-character shared-prefix heuristic to handle common inflections
    without a full stemmer, e.g.:
      "deductible" / "deduction" / "deducted"  → share "deduct" ✓
      "nominate"   / "nomination"               → share "nomin"  ✓
      "tax"        / "taxation"                 → share "tax"    ✓ (exact)

    Short topic words (< 5 chars, e.g. "term") never reach that prefix
    heuristic, so they only ever match an EXACT identical word. Tried adding
    a blanket trailing-"s" plural/singular rule to fix "term" vs "terms" —
    reverted immediately: it let the topic "term insurance" pass on any
    chunk containing the word "terms" in the totally unrelated sense of
    "terms and conditions" (a near-universal phrase in every insurance
    document, present on almost every page regardless of topic). Confirmed
    live: this let a term-insurance claim question answer confidently using
    retrieved MOTOR insurance claims content (driving license, FIR for
    vehicle theft) — a wrong-topic answer, strictly worse than the refusal
    it replaced. Exactly the "weak rider smuggles the query through" failure
    mode already documented below for the no-claim-bonus case, just via
    grammatical number instead of a generic rider word. A safe fix for the
    "term"/"terms" case needs to be specific to that word pair (or require
    the plural to appear near "insurance"/"policy" specifically), not a
    general suffix rule — left as future work rather than shipping this.
    """
    if topic == chunk_word:
        return True
    # Require at least 5 chars in both to avoid spurious short-word matches
    min_len = 5
    prefix = min(len(topic), len(chunk_word), min_len)
    return prefix >= min_len and topic[:prefix] == chunk_word[:prefix]


def _is_stopword(word: str, stop: set[str]) -> bool:
    """True if word is (or shares a root with) an entry in the stopword set.

    A hand/LLM-curated list only ever lists one inflection ("insurer") and
    silently misses others ("insurers"). Since that missed plural is common
    enough to appear in almost every KB chunk, it then slips through the
    coverage-check gate as if it were a topic-specific word, letting the LLM
    "confirm" a query that isn't actually covered by the KB. Root-matching
    against the stopword set closes this gap without hand-listing every form.
    """
    if word in stop:
        return True
    return any(_word_matches(word, s) for s in stop)


def _topics_hit_chunk(topics: set, chunk_terms: set) -> bool:
    """True if any topic word matches (exactly or by root prefix) any chunk term."""
    # Fast exact intersection first
    if topics & chunk_terms:
        return True
    # Stem-based fallback for inflection mismatches
    for t in topics:
        for c in chunk_terms:
            if _word_matches(t, c):
                return True
    return False


# Splits a compound query on "and"/"or" only when a question-word/aux-verb
# immediately follows — see the regex-fallback branch of
# _context_covers_query() for why a bare split on every "and" is wrong.
_COMPOUND_SPLIT = re.compile(
    r'\b(?:and|or)\s+(?=(?:what|who|which|how|why|when|where|is|are|'
    r'does|do|can|could|should|will|would|was|were)\b)',
    re.IGNORECASE,
)


def _context_covers_query(query: str, docs: list, llm_topics: set | None = None) -> bool:
    """True if >=1 retrieved chunk contains at least one topic keyword (root-aware).

    When llm_topics is provided, checks that every topic word appears in at
    least one retrieved chunk (distributed coverage across chunks). This is
    intentionally lenient: the LLM receives all chunks as context, so info
    spread across chunks is still usable. The only hard block is when a
    specific term (e.g. 'IDV', 'no-claim bonus') doesn't appear ANYWHERE in
    the retrieved pool, which reliably indicates the KB has no relevant content.

    Falls back to regex (OR logic) when llm_topics is absent.
    """
    if llm_topics:
        # Excludes chunks matching _REGULATORY_BOILERPLATE_RE from this max —
        # confirmed live via the hollow-answer investigation: "my claim got
        # rejected, what do I do" retrieved a regulatory/supervisory chunk
        # about IRDA's grievance-redressal and inter-party dispute-
        # adjudication framework that scored highly enough on lexical
        # similarity ("claim", "dispute") to trip the single_topic bypass
        # below on its own, letting a chunk that doesn't actually answer a
        # consumer's "why was MY claim rejected" question vouch for
        # coverage. The model then had nothing real to draw from and
        # produced a near-empty answer that technically passed every rule
        # but helped no one (see the hollow-answer detector further down —
        # that catches the SYMPTOM after generation; this is the same
        # underlying cause, caught before generation runs at all). A
        # boilerplate chunk can still be used as regular CONTEXT if it's
        # not the only thing available — this only stops it from being the
        # sole justification for skipping the stricter per-topic check.
        top_rerank = max(
            (
                d.metadata.get("rerank_score", 0) for d in docs
                if hasattr(d, "metadata")
                and not _REGULATORY_BOILERPLATE_RE.search((getattr(d, "page_content", "") or "").lower())
            ),
            default=0,
        )
        stop = _QUERY_STOP_WORDS
        topic_word_lists = []
        for topic in llm_topics:
            words = [w for w in topic.replace('-', ' ').split() if len(w) >= 3 and not _is_stopword(w, stop)]
            if words:
                topic_word_lists.append(words)

        # Score-based rescues (the 0.5 full-bypass below and the 0.05
        # lexical-miss fallback further down) are only safe for single-topic
        # questions. top_rerank is the SINGLE best score across ALL chunks —
        # for a compound question ("X and Y"), a high score usually just
        # means ONE part matched well, and applying it question-wide lets
        # that one strong match vouch for a completely unrelated other part.
        # Measured: "what is a deductible and what is the GST rate on
        # insurance premiums" scored 0.53 purely from the deductible half,
        # bypassing this check entirely and letting the model invent a
        # specific GST percentage that appears nowhere in the KB. For
        # multi-topic questions, every topic must pass the strict per-chunk
        # lexical check below — no score-based rescue.
        single_topic = len(topic_word_lists) <= 1

        # Only bypass keyword coverage check when the cross-encoder is *highly*
        # confident the chunk is relevant. The reranker (BAAI/bge-reranker-base
        # via sentence-transformers CrossEncoder) is sigmoid-activated, so
        # scores are bounded probabilities in [0, 1], NOT raw unbounded
        # logits — empirically: unrelated content ≈ 0.0000, marginally-topical
        # content ≈ 0.01-0.16, genuinely on-topic ≈ 0.3-0.5+, near-certain
        # matches ≈ 0.9+. A stale threshold of 4.0 here was calibrated for
        # raw logits and was literally unreachable (scores can't exceed 1.0),
        # silently making this bypass permanently dead code.
        if single_topic and top_rerank >= 0.5:
            return True

        # Require EVERY extracted topic to be covered (AND-logic), not just
        # one via OR-logic. The LLM extracts several "Specific" topics per
        # question, but many are weak riders that survive stopword-filtering
        # down to one common word — e.g. "motor insurance" -> "motor",
        # "health insurance" -> "health", or a country name like "india" —
        # which trivially matches almost any chunk in that domain. Under
        # plain OR-logic, a query like "what is a no-claim bonus" (topics:
        # "no-claim bonus", "motor insurance", "premiums") passed because
        # "motor" and "premiums" matched SOMETHING, even though the actual
        # distinguishing term "bonus" never appeared anywhere — the KB
        # genuinely doesn't define NCB, but a weak rider smuggled the query
        # through. An earlier attempt to fix this by checking only the
        # "most specific" (longest) topic broke when several topics tied
        # for length (e.g. "india", "health", "portability" are all single
        # words after filtering) — the arbitrary tie-break could pick the
        # weak rider instead of the real term. Requiring ALL topics to pass
        # removes the ambiguity: a weak rider passing no longer matters if
        # the real term still fails.
        #
        # Each topic's words must co-occur within a SINGLE chunk (not be
        # scattered independently across the whole retrieved pool — "zero"
        # in one unrelated chunk and "depreciation" in another otherwise
        # satisfied "zero depreciation" even though no chunk ever discussed
        # the concept as a unit) — but different topics may be satisfied by
        # different chunks, since the LLM sees the union of all of them.
        if topic_word_lists:
            chunk_term_sets = []
            for doc in docs:
                text = (
                    doc.page_content if hasattr(doc, 'page_content') else doc.get('text', '')
                ).lower()
                chunk_term_sets.append(set(re.findall(r'\b[a-z]{3,}\b', text)))
            all_topics_covered = all(
                any(
                    all(any(_word_matches(w, c) for c in chunk_terms) for w in words)
                    for chunk_terms in chunk_term_sets
                )
                for words in topic_word_lists
            )
            if all_topics_covered:
                return True

        # Lexical matching (even root-aware, even per-chunk) structurally
        # can't handle true synonyms — "claim rejected" vs. source text
        # saying "claim refused" share no root, so no hand-maintained
        # synonym list can ever be complete. But the cross-encoder reranker
        # IS a semantic model — it's what separates genuinely-relevant-but-
        # differently-worded content from actual noise, independent of
        # literal wording. When the lexical AND-check fails, fall back to
        # trusting that signal — but only well above the noise floor, and
        # only for single-topic questions (see single_topic above — this
        # fallback has the same compound-question smuggling risk as the 0.5
        # bypass): measured false positives (e.g. "IDV" scored 0.024 with
        # zero actual IDV content) sit close to measured true positives
        # (e.g. "rejected"/"refused" scored 0.074), so this bar is
        # deliberately conservative and still imperfect at this boundary —
        # a known residual limit of a purely retrieval-based gate, not a
        # fully solved problem.
        if single_topic and top_rerank >= 0.05:
            return True
        return False

    # Regex fallback (used for follow-up turns, where extracted LLM topics
    # from the raw short follow-up text aren't trustworthy — the topic lives
    # in conversation history, not the follow-up itself). Split compound
    # questions on "and"/"or", one discriminating-word set per sub-query.
    #
    # Same AND-logic + per-chunk co-occurrence fix as the llm_topics branch
    # above, and for the same reason: the old version required only ONE word
    # of a sub-query's term set to appear ANYWHERE in the pool (OR-logic,
    # scattered matching) — so "is regular health insurance mandatory"
    # (terms: "health", "mandatory") passed on "health" alone appearing
    # somewhere in an unrelated section, even though "mandatory" never
    # appeared anywhere and the KB never addresses legal requirements at
    # all. Every discriminating word of a sub-query must now co-occur
    # within a single chunk, and every sub-query must be satisfied.
    #
    # `query` here is usually the LLM-reformulated retrieval phrase for a
    # follow-up (e.g. "standard fire insurance policy exclusions and
    # limitations clause analysis"), not the user's raw question — and plain
    # "and"/"or" is extremely common in ordinary topic phrasing ("exclusions
    # and limitations", "terms and conditions") without meaning two separate
    # questions. Splitting on every bare "and" turned each such phrase into
    # two sub-queries, and the second half ("limitations clause analysis")
    # never lexically matches the actual chunks, failing the whole coverage
    # check and refusing a genuinely-covered topic. Real compound questions
    # ("what is a deductible and what is the GST rate...") repeat a
    # question-word/aux-verb right after the conjunction — only split there.
    sub_queries = [q.strip() for q in _COMPOUND_SPLIT.split(query) if q.strip()]
    all_term_sets = [_extract_topic_terms(sq) for sq in sub_queries]
    all_term_sets = [t for t in all_term_sets if t]
    if not all_term_sets:
        return True

    # Same single-topic-gated reranker-score rescue as the llm_topics branch
    # above, and it belongs here for the identical reason: pure lexical
    # matching can't handle phrasing gaps ("proximate cause" query vs.
    # source explaining it via "nearest cause"/"closest cause" without ever
    # using the words "proximate" or "determined" together) — the LLM-topics
    # extraction can also come back empty for a legitimate question (observed:
    # "How is proximate cause determined when multiple causes contribute to a
    # loss?" returned zero specific topics), which silently routes here with
    # no fallback at all before this fix, even when the actual best chunk
    # scored 0.92 — about as confident as this system ever gets.
    top_rerank = max(
        (d.metadata.get("rerank_score", 0) for d in docs if hasattr(d, "metadata")),
        default=0,
    )
    single_topic = len(all_term_sets) <= 1
    if single_topic and top_rerank >= 0.5:
        return True

    chunk_term_sets = []
    for doc in docs:
        text = (
            doc.page_content if hasattr(doc, 'page_content') else doc.get('text', '')
        ).lower()
        chunk_term_sets.append(set(re.findall(r'\b[a-z]{3,}\b', text)))

    all_covered = all(
        any(
            all(any(_word_matches(w, c) for c in chunk_terms) for w in term_set)
            for chunk_terms in chunk_term_sets
        )
        for term_set in all_term_sets
    )
    if all_covered:
        return True

    if single_topic and top_rerank >= 0.05:
        return True
    return False


def _quoted_comparison_covered(question: str, docs: list) -> bool:
    """True unless the question quotes 2+ specific terms to compare and the
    retrieved context is missing one of them.

    _context_covers_query() only requires ANY topic word to appear ANYWHERE
    in the retrieved pool, so a question like 'difference between "floater
    policy" and "specific policy" in fire insurance' passes on the generic
    words "fire"/"policy" alone even when the KB never discusses a floater
    vs. specific split — the LLM then fills the gap from training knowledge.
    This closes that gap: every quoted term must have at least one of its
    non-generic words present in context, or we decline instead of guessing.
    """
    quoted = re.findall(r'["‘’“”]([^"‘’“”]{3,40})["‘’“”]', question)
    quoted = [q.strip() for q in quoted if q.strip()]
    if len(quoted) < 2:
        return True
    ctx_lower = " ".join(
        (doc.page_content if hasattr(doc, "page_content") else doc.get("text", ""))
        for doc in docs
    ).lower()
    for term in quoted:
        distinguishing = _extract_topic_terms(term)
        if not distinguishing:
            continue
        if not any(w in ctx_lower for w in distinguishing):
            return False
    return True


# "Which insurers cover X" / "who offers Y" expect a NAMED list back.
# _context_covers_query() only requires generic topic words ("travel",
# "america") to appear somewhere in the retrieved pool — a general
# educational video that mentions America as an example destination passes
# that gate even though it never names a single insurer. This closes that
# gap: enumeration-style questions additionally require at least one
# proper-noun-like token (a plausible brand/company name) in the context.
_ENUMERATION_PATTERN = re.compile(
    r'\b(which|what)\s+(\w+\s+){0,2}(insurers?|insurance\s+compan(?:y|ies)|'
    r'compan(?:y|ies)|providers?|banks?|firms?|carriers?)\b'
    r'|\bwho\s+(covers?|offers?|provides?|insures?|sells?|underwrites?)\b'
    r'|\b(name|list)\s+(the|all|some)\b.{0,30}(insurers?|compan(?:y|ies)|providers?)',
    re.IGNORECASE,
)

_ENUM_COMMON_CAP_WORDS = {
    "the", "this", "that", "these", "those", "america", "india", "usa", "uk",
    "europe", "asia", "africa", "australia", "canada", "china", "japan",
    "however", "therefore", "additionally", "furthermore", "generally",
    "typically", "importantly", "note", "tip", "example", "for", "when",
    "while", "before", "after", "during", "also", "some", "many", "most",
    "several", "various", "certain", "so", "now", "here", "there", "yes", "no",
    # Secondary safeguard (belt-and-suspenders, not the primary fix — see
    # _has_named_entity()): these are capitalized when they open a
    # proper-noun-style phrase like "Takaful Insurance" or "Islamic
    # insurance model", but none of them are company names. The primary
    # fix is requiring known-insurer-name or company-suffix-adjacency
    # evidence in _has_named_entity() itself; this list additionally
    # excludes these specific words even if they'd otherwise satisfy that
    # adjacency check (e.g. "Islamic insurance" — "Islamic" sits directly
    # next to the suffix word "insurance").
    "takaful", "shariah", "islamic", "halal", "conventional",
}

# Known insurer names — factored out from _is_likely_followup()'s
# _INSURANCE_INDICATORS_LOCAL regex (same list) below, so both places share
# one source of truth instead of maintaining two independent copies.
_KNOWN_INSURER_NAMES = frozenset({
    "hdfc ergo", "icici", "bajaj", "tata aig", "reliance",
    "new india", "oriental", "national", "united india",
})

# Capitalized word/phrase immediately adjacent to one of these counts as a
# plausible company name (e.g. "New India Assurance", "XYZ Ltd", "ABC Corp").
_COMPANY_SUFFIX_WORDS = frozenset({
    "insurance", "assurance", "general", "life", "ltd", "inc", "co",
    "corp", "group", "underwriters", "mutual", "holdings",
})


def _has_named_entity(text: str) -> bool:
    """Plausible company/brand name: either a known insurer name, or a
    capitalized word/phrase immediately adjacent to a company-suffix word.

    The old version treated ANY capitalized, non-sentence-initial word not
    in a small filler list as a "plausible company name" — which let words
    like "Takaful", "Islamic", "Shariah" (capitalized when they open a
    proper-noun-style phrase, e.g. "Takaful Insurance") pass as named
    entities. That wrongly satisfied _enumeration_query_covered() for
    "which insurers cover the travel policy to south africa" when the
    retrieved chunk was actually about the Takaful insurance MODEL and
    never named a single real insurer — the confident answer that reached
    the user cited that chunk as if it answered the question.
    """
    lower = text.lower()
    if any(name in lower for name in _KNOWN_INSURER_NAMES):
        return True
    sentence_initial = {
        m.start(1) for m in re.finditer(r'(?:^|[.!?]\s+)([A-Z][a-zA-Z]{2,})', text)
    }
    for m in re.finditer(r'\b[A-Z][a-zA-Z]{2,}\b', text):
        if m.start() in sentence_initial:
            continue
        word = m.group(0)
        if word.lower() in _ENUM_COMMON_CAP_WORDS:
            continue  # belt-and-suspenders exclusion — see _ENUM_COMMON_CAP_WORDS
        before = text[:m.start()].rstrip()
        after = text[m.end():].lstrip()
        before_word = re.search(r'(\w+)$', before)
        after_word = re.match(r'^(\w+)', after)
        if before_word and before_word.group(1).lower() in _COMPANY_SUFFIX_WORDS:
            return True
        if after_word and after_word.group(1).lower() in _COMPANY_SUFFIX_WORDS:
            return True
    return False


def _enumeration_query_covered(question: str, docs: list) -> bool:
    """True unless the question asks 'which/who provides X' and the
    retrieved context contains no plausible named entity at all — or has a
    named entity, but not in the SAME chunk as the question's own topic
    words. A named entity that only appears in an unrelated chunk doesn't
    answer "which insurers cover travel to south africa" just because SOME
    insurer name exists somewhere else in the retrieved pool.
    """
    if not _ENUMERATION_PATTERN.search(question):
        return True
    topic_words = _extract_topic_terms(question)
    for doc in docs:
        text = doc.page_content if hasattr(doc, "page_content") else doc.get("text", "")
        if not _has_named_entity(text):
            continue
        if not topic_words:
            # No specific topic words to check co-occurrence against (e.g.
            # question was just "which insurers offer coverage?") — a named
            # entity anywhere is sufficient, same as before.
            return True
        chunk_terms = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
        if _topics_hit_chunk(topic_words, chunk_terms):
            return True
    return False


_DETAIL_SIGNALS = {
    'in detail', 'step by step', 'step-by-step', 'in steps', 'procedure',
    'how to', 'elaborate', 'walk me through',
    'in depth', 'thoroughly', 'fully explain', 'all about', 'everything about',
    'comprehensive', 'what are all', 'list all',
    'what all', 'how does it work', 'all the steps', 'entire process',
    'tell me everything', 'give me a full', 'give me the full',
}

_SIMPLE_SIGNALS = {
    'in simple', 'in short', 'in brief', 'briefly', 'short answer',
    'simple answer', 'quick answer', 'simple terms', 'layman', 'easy way',
    'just briefly', 'keep it short', 'keep it simple', 'just tell me',
    'just give me', 'just explain', 'just say',
    'simple language', 'simple words', 'easy language', 'easy words',
    'meaning of', 'meaning of this', 'what do you mean',
}

_EXAMPLE_SIGNALS = {
    'with example', 'with an example', 'give me an example', 'give an example',
    'show me an example', 'show an example', 'can you give example',
    'explain with example', 'example please', 'real life example',
    'real example', 'for example', 'use example',
    'with the help of an example', 'with the help of example',
    'explain me with example', 'can you give me an example',
    'explain with the help', 'using an example', 'using example',
}

# Every entry above contains the word "example" — the phrase list is really a
# proxy for "does the user want one". A fixed phrase list breaks on ordinary
# grammar drift (e.g. "with a example" instead of "with an example"), which
# silently disabled both the KV-cache safeguard and the prompt instruction for
# that query. Match on the word itself instead, which is the real signal.
_EXAMPLE_PATTERN = re.compile(r'\bexamples?\b', re.IGNORECASE)


def _wants_example(text: str) -> bool:
    return bool(_EXAMPLE_PATTERN.search(text))


# Same fragility as _EXAMPLE_SIGNALS had: _DETAIL_SIGNALS' 'in detail' entry
# is an exact-phrase match, so ordinary grammar drift silently disables it.
# Confirmed live: "can you explain it in more detail" — an extremely common,
# natural follow-up phrasing — doesn't contain the literal substring "in
# detail" (the word "more" sits between "in" and "detail"), so it silently
# fell through to brief mode. The user got a marginally-longer paragraph
# instead of the structured, wider-retrieval DETAILED_GROUNDED_PROMPT they
# asked for — visible from the outside only as the answer keeping brief
# mode's sign-off ("Let me know if you want more details!") instead of
# detailed mode's ("Hope that clears it up!..."). Same fix shape as
# _EXAMPLE_PATTERN: match the core signal via regex instead of an exact
# phrase, permissive of 0-2 modifier words between "in" and "detail(s)" so
# "in detail", "in more detail", "in greater detail", "in much more detail"
# all match.
_DETAIL_PATTERN = re.compile(r'\bin\s+(?:\w+\s+){0,2}details?\b', re.IGNORECASE)


def _needs_detailed_answer(question: str) -> bool:
    """True when the question expects a comprehensive, multi-part, or procedural answer."""
    q = question.lower()
    if any(sig in q for sig in _SIMPLE_SIGNALS):
        return False  # user explicitly wants brief — short-circuit before detail check
    if any(sig in q for sig in _DETAIL_SIGNALS) or _DETAIL_PATTERN.search(q):
        return True
    # Long questions (>25 words) almost always need more than 4 sentences
    if len(question.split()) > 25:
        return True
    # Three or more sub-questions joined by "and"
    if q.count(' and ') >= 3:
        return True
    return False


# ── Hybrid fallback for detail/simple/example intent ───────────────────────
# _SIMPLE_SIGNALS/_DETAIL_SIGNALS/_wants_example above are a fast, free,
# zero-latency check that correctly classifies the vast majority of real
# phrasings — but a fixed phrase/regex list can only ever cover phrasings
# someone has already hit (see [[project_detail_pattern_generalization]],
# [[project_fragile_signal_lists]]: EXAMPLE and DETAIL were each patched
# after a live miss, and there will always be a next one no matter how many
# patterns get added). The genuinely general fix is to ask the LLM itself
# when the fast path is inconclusive — semantic understanding generalizes to
# any phrasing, lexical matching never fully does.
#
# Kept as a FALLBACK, not the primary path, specifically for latency: this
# project has repeatedly traded effort for latency elsewhere this session
# (consolidated reranking, sentence caching, suggestion-chip removal), so
# adding a blocking classification call to every single request would cut
# directly against that. The fast path already resolves ~all real traffic;
# the LLM call only fires for the (expected to be rare) case where NONE of
# detail/simple/example matched anything lexically, so most requests pay
# zero extra latency and only genuinely ambiguous/novel phrasing pays the
# ~1-2s round trip.
_MODIFIER_INTENT_PROMPT = """\
Read the user's message to an insurance chatbot and decide what kind of \
answer they want.

DETAIL = the message EXPLICITLY asks for a full, comprehensive, step-by-\
step, or thorough explanation — not just a plain question.
SIMPLE = the message EXPLICITLY asks for a brief, short, plain-language, \
or easy-to-understand answer — not just a plain question.
EXAMPLE = the message EXPLICITLY asks for a concrete example, \
illustration, or real-life scenario.

A plain, ordinary question ("What is X?", "How does X work?", "Does X \
cover Y?") with NO explicit request for a particular style or depth is \
"no" on all three — default to "no" unless the wording clearly asks for \
one of these. Do not infer DETAIL or SIMPLE from how long or short the \
question itself is, or guess that a basic question implicitly wants a \
short answer — many basic questions still want a normal, complete answer.

Any, all, or none of these can be true for a given message. Output EXACTLY \
one line, nothing else, in this exact format:
detail=<yes/no> simple=<yes/no> example=<yes/no>

Message: "What is health insurance?"
detail=no simple=no example=no

Message: "I want the full picture, not just a summary"
detail=yes simple=no example=no

Message: "keep it short please"
detail=no simple=yes example=no

Message: "can you show me a real scenario for this"
detail=no simple=no example=yes

Message: {question}
"""


# Loose, cheap pre-filter for whether the LLM fallback is even worth
# trying. Confirmed live this matters: without it, "What is fire
# insurance?" — a completely plain question with zero modifier-related
# vocabulary — fell through to the LLM fallback exactly like a genuine
# novel-phrasing case, because "the strict fast path matched nothing" is
# also true for the overwhelming majority of ordinary questions that were
# never asking for a modifier at all. Not matching the strict signal list
# means "resolved: no modifier" for a plain question; it only means
# "inconclusive, worth asking" when the wording ALSO contains some
# modifier-adjacent vocabulary the strict list didn't happen to cover.
# This word set is deliberately looser/broader than the strict signal
# lists — it only has to be a cheap, in-process membership check, not an
# accurate classifier; the LLM call that follows is the actual classifier.
_MODIFIER_HINT_WORDS = {
    'full', 'complete', 'thorough', 'thoroughly', 'comprehensive',
    'everything', 'all', 'deep', 'deeper', 'depth', 'elaborate', 'expand',
    'picture', 'scenario', 'illustrate', 'breakdown', 'overview',
    'summary', 'summarize', 'concise', 'brief', 'briefly', 'quick',
    'short', 'basics', 'layman', 'plain', 'simple', 'simplify',
    'detail', 'details', 'detailed', 'skip',
}
# Multi-word hints can't be caught by single-word tokenization above —
# checked separately as plain substrings.
_MODIFIER_HINT_PHRASES = ('hold back', 'leave out', 'skip ahead')


def _has_modifier_hint(question_lower: str) -> bool:
    tokens = set(re.findall(r"[a-z]+(?:'[a-z]+)?", question_lower))
    if tokens & _MODIFIER_HINT_WORDS:
        return True
    return any(phrase in question_lower for phrase in _MODIFIER_HINT_PHRASES)


async def _classify_modifier_intent_llm(question: str) -> tuple[bool, bool, bool]:
    """LLM fallback for detail/simple/example intent — only called when the
    fast keyword/regex check found nothing in any of the three dimensions.
    max_tokens=20 so the round trip stays short. Falls back to
    (False, False, False) on any error/timeout, same as the fast path
    finding nothing — the caller already treats that as "no modifier
    requested", so a failed classification call degrades to the pre-hybrid
    behavior rather than blocking or crashing the request.
    """
    try:
        prompt = _MODIFIER_INTENT_PROMPT.format(question=question)
        raw = await _backend_completion(prompt, max_tokens=20, timeout=4)
        if not raw:
            return (False, False, False)
        raw_l = raw.lower()
        return (
            bool(re.search(r"detail\s*=\s*yes", raw_l)),
            bool(re.search(r"simple\s*=\s*yes", raw_l)),
            bool(re.search(r"example\s*=\s*yes", raw_l)),
        )
    except Exception:
        return (False, False, False)


async def _resolve_modifier_intent(question: str) -> tuple[bool, bool, bool]:
    """Resolve whether the user wants a detailed, simple, and/or
    example-based answer. Three stages: (1) strict fast path (free) — a
    real phrase/regex match resolves immediately; (2) if that finds
    nothing, a loose hint-word pre-filter decides whether the question is
    even worth asking the LLM about — a genuinely plain question with no
    modifier-adjacent vocabulary at all resolves to "no modifier" here,
    still free; (3) only if the loose filter also finds something does the
    LLM classification call actually run. See the block comment above
    _MODIFIER_INTENT_PROMPT for why this is hybrid rather than
    always-fast or always-LLM, and _has_modifier_hint's comment for why
    stage 2 exists (a plain "What is fire insurance?" was paying the LLM
    round trip for nothing until stage 2 was added).

    Returns (has_detail, has_simple, has_example). SIMPLE takes precedence
    over DETAIL when both are somehow set, matching _needs_detailed_answer's
    existing precedence — a user asking for both is treated as wanting
    brief, not a contradiction to resolve either way.
    """
    q = question.lower()
    _fast_example = _wants_example(q)
    _fast_simple = any(sig in q for sig in _SIMPLE_SIGNALS)
    _fast_detail = any(sig in q for sig in _DETAIL_SIGNALS) or bool(_DETAIL_PATTERN.search(q))
    # Structural signals (long/compound questions) stay fast-path-only —
    # these aren't a phrasing-recognition gap the LLM classifier would help
    # with, they're "is this objectively a big question" regardless of
    # wording, already phrasing-independent by construction.
    _structural_detail = len(question.split()) > 25 or q.count(' and ') >= 3

    if _fast_example or _fast_simple or _fast_detail or _structural_detail:
        _has_detail = (_fast_detail or _structural_detail) and not _fast_simple
        return (_has_detail, _fast_simple, _fast_example)

    # Strict fast path matched nothing — for the overwhelming majority of
    # ordinary questions ("What is fire insurance?") that's not ambiguity,
    # it's the correct, confident answer: no modifier was requested. Only
    # worth the LLM round trip when the wording ALSO contains some looser
    # modifier-adjacent vocabulary the strict list didn't happen to cover —
    # otherwise every plain question would pay the fallback's latency for
    # no reason, defeating the entire point of the hybrid design.
    if not _has_modifier_hint(q):
        return (False, False, False)

    # Genuinely ambiguous or novel phrasing (e.g. "I want the full
    # picture, not just a summary", or "explain it in more detail" before
    # that specific gap was patched into the strict list). Try the cheap
    # LLM classifier before defaulting to "no modifier requested".
    _llm_detail, _llm_simple, _llm_example = await _classify_modifier_intent_llm(question)
    _has_detail = _llm_detail and not _llm_simple
    logger.info(
        "[ask_stream] modifier intent LLM fallback fired: detail=%s simple=%s example=%s query=%r",
        _has_detail, _llm_simple, _llm_example, question[:80],
    )
    return (_has_detail, _llm_simple, _llm_example)


_FOLLOWUP_SIGNALS = {
    'it', 'its', 'that', 'this', 'them', 'those', 'they', 'their', 'which', 'these',
}

# Pronoun/reference words used to decide whether _contextualize_query() should
# attempt an LLM call.  If the question doesn't contain any of these tokens it
# is structurally standalone (e.g. a fresh topic out of nowhere), so the LLM
# round-trip can be skipped entirely.
#
# Ordinal references to a numbered list ("point 2", "the 2nd point", "explain
# number 3") are just as much a reference needing resolution as "the second
# point" — but people type them with a digit far more often than spelled out.
# Without the digit patterns below, "explain the 2 point" fell through this
# fast-path as "structurally standalone" and skipped contextualization
# entirely, so the follow-up went to retrieval completely unresolved and the
# model ended up guessing/blending a different point at generation time.
_REFERENCE_TOKENS = re.compile(
    r"\b(?:it|its|that|this|those|these|they|them|their|which|"
    r"one\b|ones\b|the\s+\w+\s+one|the\s+other\b|"
    r"more\b|further\b|elaborate\b|"
    r"first\b|second\b|third\b|last\b|"
    r"other\b|another\b|"
    r"\d+(?:st|nd|rd|th)\b|"
    r"point\s+\d+\b|\d+\s*(?:st|nd|rd|th)?\s+point\b|"
    r"number\s+\d+\b)",
    re.IGNORECASE,
)
_FOLLOWUP_OPENERS = (
    'what about', 'how about', 'and what', 'also ', 'tell me more',
    'what does it', 'how does it', 'what is it', 'is it ', 'can it ',
)

# Standalone keywords that strongly signal a follow-up when the message is short
# and has no standalone insurance topic. These words refer back to a prior topic
# (e.g. "justify more", "explain further", "why though") rather than introducing
# a new self-contained question.
_FOLLOWUP_KEYWORDS = frozenset({
    "more", "further", "why", "justify", "expand", "deeper", "elaborate",
    "explain", "details", "detail", "reason", "reasons", "example", "examples",
    "though", "still", "again", "anyway",
})


async def _backend_completion(
    prompt: str, max_tokens: int, timeout: float, temperature: float = 0,
    backend_override: Optional[str] = None,
) -> Optional[str]:
    """Fast, non-streaming chat completion against whichever backend is
    currently active (vLLM or Groq — both OpenAI-compatible, same request
    shape). Returns the raw response text, or None on any failure, timeout,
    or unconfigured/unsupported backend.

    Used for short, best-effort auxiliary calls (topic extraction, query
    reformulation) that should follow the SAME backend as the main
    answer generation, not be pinned to vLLM regardless of FORCE_BACKEND.
    An earlier version hardcoded these to
    always use vLLM specifically to keep a Groq-vs-vLLM generation-fidelity
    A/B test uncontaminated by a different topic-extraction model also
    changing retrieval-gating behavior. That trade-off made sense for a
    controlled test, but not for actual production use — once FORCE_BACKEND
    picks a backend to actually serve answers from, every supporting call
    should use that same fast backend too, not bottleneck the whole request
    on a slower one just for a few background tokens.

    backend_override bypasses _active_backend() for the one call that needs
    it regardless of FORCE_BACKEND: the query-cleaning fallback path exists
    specifically to route around Groq's daily quota running out (observed
    repeatedly in testing), so its own grounding re-check would defeat the
    point if it silently used Groq again and failed the same way.

    Only vLLM and Groq are handled (matches what existed before this
    generalization) — OpenAI/Anthropic backends fall through to None here,
    same as an unconfigured backend, and callers already have a graceful
    fallback for that.
    """
    import aiohttp as _ah
    from router import (
        VLLM_HOST, VLLM_API_KEY, _resolve_vllm_model, _active_backend,
        GROQ_API_KEY, GROQ_MODEL,
    )
    backend = backend_override or _active_backend()
    if backend == "vllm":
        url = f"{VLLM_HOST}/v1/chat/completions"
        model = _resolve_vllm_model()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {VLLM_API_KEY}"}
    elif backend == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
        model = GROQ_MODEL
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROQ_API_KEY}"}
    elif backend == "manual":
        from router import _runtime_manual_api_key, _runtime_manual_base_url, _runtime_manual_model
        url = f"{_runtime_manual_base_url.rstrip('/')}/chat/completions"
        model = _runtime_manual_model
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {_runtime_manual_api_key}"}
    else:
        return None
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    try:
        async with _ah.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers,
                timeout=_ah.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status != 200:
                    # Silently returning None here (as before) is indistinguishable
                    # from a network blip or timeout in logs — every caller treats
                    # it as "the model said no" and refuses, so a sustained
                    # condition (e.g. a rate/quota limit) looks identical to
                    # ordinary occasional flakiness with no way to tell them apart
                    # short of manually replaying the exact request outside the
                    # app. Logging status + body once per failure costs nothing
                    # on the happy path and turns "everything is mysteriously
                    # refusing" into an immediately diagnosable log line.
                    body = await resp.text()
                    logger.warning(
                        "[_backend_completion] %s returned status=%s (backend=%s model=%s): %s",
                        url, resp.status, backend, model, body[:300],
                    )
                    return None
                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        logger.warning("[_backend_completion] request to %s failed (backend=%s): %s", url, backend, exc)
        return None


async def _vllm_clean_query(question: str) -> Tuple[Optional[str], Optional[str], bool]:
    """Fallback query cleaner tried only when the primary retrieval attempt
    already failed coverage — fixes general spelling mistakes (not just the
    hand-curated insurance vocabulary _correct_typos() knows) and strips
    specific qualifiers (a named country/city, emphasis words like
    "affordable"/"cheap"/"best") that measurably tank the reranker's score
    even when the KB covers the general topic well. Measured: "how to buy
    motor insurance" scored 0.79; adding just "affordable" dropped it to
    0.017; adding "in Dubai" alone dropped it to 0.15 — the same qualifier-
    dilution pattern behind several earlier fixes this session, just never
    addressed at the query-cleaning layer directly.

    Calls vLLM specifically, not whatever _active_backend()/FORCE_BACKEND
    currently points at — this is a retry path for a query that's already
    failing, not primary answer generation, and shouldn't compete for the
    same (rate-limited) Groq quota real answers need.

    Returns (cleaned_query, dropped_terms_summary, dropped_has_proper_noun)
    — cleaned_query and dropped_terms_summary are both None on any failure,
    unconfigured backend, or empty response, so the caller falls through to
    the existing refusal exactly as it did before this function existed.
    dropped_terms_summary is computed programmatically from the before/
    after diff (see _diff_dropped_terms), not self-reported by the model —
    asking it to also report what it removed was tried first and was
    unreliable, sometimes echoing the prompt's own example words
    ("affordable", "cheap", "best") as "dropped" even when they were never
    in the query, or weren't actually removed from the cleaned version.
    """
    import aiohttp as _ah
    from router import VLLM_HOST, VLLM_API_KEY, _resolve_vllm_model
    if not VLLM_HOST:
        return None, None, False
    prompt = (
        f"Query: {question}\n\n"
        "This insurance-related search query may contain spelling mistakes, "
        "and words that make it too specific for a keyword search to match "
        "well (e.g. a country/city name, or emphasis words like "
        "'affordable', 'cheap', 'best'). Rewrite it as a short, general "
        "search query about the CORE topic only: fix any spelling mistakes, "
        "and drop any specific location or emphasis words that aren't "
        "needed to describe the core insurance topic. "
        "Respond with ONLY the rewritten query, nothing else."
    )
    try:
        url = f"{VLLM_HOST}/v1/chat/completions"
        model = _resolve_vllm_model()
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {VLLM_API_KEY}"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 40,
            "temperature": 0,
            "stream": False,
        }
        async with _ah.ClientSession() as session:
            async with session.post(
                url, json=payload, headers=headers,
                timeout=_ah.ClientTimeout(total=12),
            ) as resp:
                if resp.status != 200:
                    return None, None, False
                data = await resp.json()
                raw = data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None, None, False
    cleaned = raw.strip().strip('"').strip("'").strip()
    if not cleaned:
        return None, None, False
    _dropped_note, _dropped_proper_noun = _diff_dropped_terms(question, cleaned)
    return cleaned, _dropped_note, _dropped_proper_noun


# Emphasis/qualifier words worth calling out by name if the cleaner drops
# them — deliberately short and specific rather than a broad stopword list,
# since the goal is only to name concrete things that got removed, not to
# flag every function word that naturally differs between two phrasings of
# the same query.
_EMPHASIS_WORDS = frozenset({
    'affordable', 'cheap', 'cheapest', 'best', 'top', 'premium',
    'expensive', 'discount', 'discounted', 'lowest', 'cheaply',
})


def _diff_dropped_terms(original: str, cleaned: str) -> Tuple[Optional[str], bool]:
    """Return (short comma-separated list of concrete terms present in
    *original* but missing from *cleaned*, whether any of them is a likely
    proper noun) — specifically capitalized words other than the first
    (likely a proper noun: a country, city, or provider name) and known
    emphasis words (_EMPHASIS_WORDS). Computed directly from the text
    rather than trusted from the model's own self-report (see
    _vllm_clean_query's docstring for why).

    The proper-noun flag matters downstream: dropping "affordable"/"cheap"
    to get a retrieval match is harmless (the KB's content on the core
    topic is still factually complete, just not tagged "cheap"), but
    dropping a country/city/provider name means the KB has no actual
    content about that specific thing — answering with the generic
    version and hoping the model remembers to disclose the gap is not
    reliable enough to trust (see the caller in ask_stream for the
    concrete failure this was measured against).
    """
    orig_words = original.split()
    clean_lower = cleaned.lower()
    dropped = []
    has_proper_noun = False
    for i, w in enumerate(orig_words):
        w_clean = w.strip(".,!?;:()[]{}'\"")
        if not w_clean:
            continue
        w_lower = w_clean.lower()
        is_proper_noun = i > 0 and w_clean[0].isupper()
        is_emphasis = w_lower in _EMPHASIS_WORDS
        if (is_proper_noun or is_emphasis) and w_lower not in clean_lower:
            dropped.append(w_clean)
            if is_proper_noun:
                has_proper_noun = True
    return (", ".join(dict.fromkeys(dropped)) if dropped else None, has_proper_noun)


_SPECIFIC_TYPE_RE = re.compile(
    r"\b(motor|car|vehicle|auto|bike|two.wheeler|four.wheeler|"
    r"life|term\s*life|whole\s*life|endowment|ulip|"
    r"health|medical|hospital|"
    r"travel|trip|flight|baggage|"
    r"home|house|property|landlord|tenant|"
    r"marine|cargo|fire|"
    r"liability|third.party|"
    r"critical\s*illness|cancer|"
    r"group|corporate|reinsurance|takaful|"
    r"crop|agricultur\w*|"
    r"personal\s*accident|disability|"
    r"retirement|pension|annuity)\b"
)

# Narrower than _SPECIFIC_TYPE_RE above: requires the type word to sit right
# next to "insurance/policy/assurance/cover" before counting as a genuine
# type ATTRIBUTION. _SPECIFIC_TYPE_RE's bare-word matching is fine for its
# original purpose (does a raw USER QUESTION name its own topic?), but reused
# against GENERATED ANSWER prose it over-fires on ordinary English: a
# correct, well-grounded takaful-insurance answer explaining "risks are
# shared among a group, rather than being transferred to an insurance
# company" got discarded and replaced with a refusal because "group" alone
# matched — the answer never claimed anything about Group Insurance as a
# product, it just used "group" to mean "a group of people." Words like
# group/life/home/fire/crop are common English outside insurance too;
# requiring adjacency to insurance/policy/assurance/cover keeps the
# genuine-attribution cases ("endowment assurance policy", "group
# insurance") while dropping incidental non-insurance uses of the same word.
_TYPE_ATTRIBUTION_RE = re.compile(
    r"\b(motor|car|vehicle|auto|bike|two.wheeler|four.wheeler|"
    r"life|term\s*life|whole\s*life|endowment|ulip|"
    r"health|medical|hospital|"
    r"travel|trip|flight|baggage|"
    r"home|house|property|landlord|tenant|"
    r"marine|cargo|fire|"
    r"liability|third.party|"
    r"critical\s*illness|cancer|"
    r"group|corporate|reinsurance|takaful|"
    r"crop|agricultur\w*|"
    r"personal\s*accident|disability|"
    r"retirement|pension|annuity)\s+(insurance|policy|assurance|cover|coverage)\b",
    re.IGNORECASE,
)

# Module-level (not request-scoped) so both the detailed-mode point filter
# and the brief-mode whole-reply check below can share one definition rather
# than drifting out of sync. Catches jargon/content from a RETRIEVED chunk
# about the wrong topic — distinct from _TYPE_ATTRIBUTION_RE above, which
# only catches an explicit "X insurance/policy" naming, not topic-specific
# vocabulary that never says the type name at all (e.g. "driving history"
# never says "motor insurance").
_TYPE_GIVEAWAY_TERMS = {
    "marine": ("marine insurance", "marine cargo", "icc (a)", "icc (b)", "institute cargo clause", "bill of lading", "voyage policy", "hull insurance"),
    "health": ("health insurance", "domiciliary hospitalization", "domiciliary hospitalisation", "cashless hospitalization", "cashless hospitalisation", "pre-existing disease"),
    "crop": ("crop insurance", "agriculturist", "crop failure", "sowing/planting", "loanee"),
    "fidelity": ("fidelity insurance", "employee dishonesty", "embezzlement"),
    "transit": ("transit insurance", "import covers by sea", "inland transit clause"),
    "motor": (
        "motor insurance", "vehicles plying on public roads",
        "vehicle", "owner-driver", "driving record", "driving history",
        "reckless driving", "unnamed passengers",
        "tp (third-party) premium", "own damage",
    ),
}
# A query naming the type itself is exempt from that type's filter (a
# genuine comparison/vehicle-linked question should keep vehicle content).
# "motor" additionally exempts on "vehicle"/"car"/"driving" etc. since a
# question can ask about vehicle-linked cover without ever saying "motor".
_TYPE_QUERY_EXEMPT_WORDS = {
    "motor": ("motor", "vehicle", "car", "bike", "scooter", "motorcycle", "driving", "driver"),
}

# A point naming a sibling policy type as somewhere ELSE a loss is covered
# ("this is excluded here; covered under your motor/fire policy instead")
# or explicitly as an exclusion is legitimate, standard insurance-document
# language, not evidence the point is actually ABOUT that sibling type.
_EXCLUSION_LANGUAGE_RE = re.compile(
    r"\bexclu\w*\b|\bnot\s+covered\b|\bdoes\s+not\s+cover\b|"
    r"\bcovered\s+(?:by|under|elsewhere)\b",
    re.IGNORECASE,
)


def _text_has_giveaway_contamination(
    text: str, query_lower: str, query_policy_type: str = "general",
    retrieved_context_text: str = "",
) -> bool:
    # query_policy_type is classify_query_policy_type(retrieval_query) —
    # already computed once per request for the metadata retrieval filter,
    # and far more reliable than re-deriving "does the query mention this
    # type" via a bare-word substring check. Confirmed live: "What is
    # medical insurance?" classifies correctly as "health" (its regex list
    # includes "medical insurance"), but the OLD bare-word exemption only
    # checked for the literal word "health" in the query — absent here — so
    # a fully correct answer ("Medical insurance is a type of health
    # insurance...") got flagged as contaminated for saying "health
    # insurance", its own genuinely-correct category. Trusting the
    # classifier directly closes this whole class of synonym/hypernym gap,
    # not just this one instance.
    #
    # retrieved_context_text requires the giveaway TERM to genuinely appear
    # somewhere in what was retrieved this turn before treating a match in
    # the generated text as contamination — same "real evidence" guarantee
    # the original design wanted, but checked against the actual retrieved
    # TEXT rather than a chunk's policy_type TAG. The tag-based version of
    # this check (`_type_name not in retrieved_types`, retrieved_types
    # built by excluding policy_type=="general") was structurally blind to
    # any topic with no official policy_type of its own — confirmed live:
    # "Explain engineering insurance in detail" retrieved ONLY
    # general-tagged chunks (engineering isn't one of the 12 hardcoded
    # types) and produced a point containing "hull insurance" — a term
    # already tracked under the marine bucket — but the marine check never
    # ran at all, since no chunk was TAGGED marine, even though the term
    # was sitting right there in the retrieved text. Checking the text
    # directly closes that gap without weakening the original guarantee:
    # a term still only counts as evidence when it's actually present in
    # what was retrieved, not just because the query's classifier guessed
    # a sibling type (e.g. explaining "personal insurance" by listing it
    # alongside motor/health as one of five general categories stays
    # exempt, since neither "motor insurance" nor "health insurance" as a
    # literal term needs to appear in the retrieved text for that to be a
    # legitimate answer).
    text_lower = text.lower()
    context_lower = retrieved_context_text.lower()
    for _type_name, _terms in _TYPE_GIVEAWAY_TERMS.items():
        if _type_name == query_policy_type:
            continue
        if not any(term in context_lower for term in _terms):
            continue
        _exempt_words = _TYPE_QUERY_EXEMPT_WORDS.get(_type_name, (_type_name,))
        if any(w in query_lower for w in _exempt_words):
            continue
        if any(term in text_lower for term in _terms):
            # A point NAMING a sibling type isn't automatically ABOUT that
            # type — a standard coordination-of-benefits/anti-duplication
            # exclusion clause ("X is excluded here because it's covered
            # under your motor policy") is completely normal, correct
            # insurance-document language, and its own sibling-type
            # classification is identical either way (both come back
            # "motor" — type confirmation, which fixed the Phase 2 gate's
            # false positives, does NOT separate this case; the giveaway
            # term itself is the same word whether the point is about that
            # type or just redirecting to it). Confirmed live 2026-07-23: a
            # correct jewellery-exclusions point naming "motor insurance
            # policies" as where an overlapping loss is instead covered
            # got deleted wholesale. Same negation-style guard the
            # fines-claim check already uses below, applied here too.
            if _EXCLUSION_LANGUAGE_RE.search(text_lower):
                continue
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# Shared numbered-point split/rebuild — contamination root-cause plan
# (plan.md), Phase 0/2. Previously this exact opener/points/closer
# split-and-fold logic was duplicated near-identically in the ungrounded-
# point filter and the cross-topic contamination filter below (and would
# have needed a THIRD copy for the point-relevance gate) — precisely the
# kind of silent-drift risk the plan calls out. One shared implementation
# now backs all three call sites plus the contamination trace.
# ─────────────────────────────────────────────────────────────────────────
_POINT_LINE_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")


def _split_numbered_points(text: str) -> Tuple[List[str], List[str], List[str]]:
    """Split numbered-list answer text into (opener_lines, point_texts,
    closer_lines). Folds indented or "-"/"*"/"•" continuation lines into
    their parent point so a later dropped point doesn't leave orphaned
    sub-bullets behind — same behavior the two original inline copies
    had, just no longer duplicated.
    """
    lines = text.split("\n")
    opener_lines: List[str] = []
    point_texts: List[str] = []
    closer_lines: List[str] = []
    seen_point = False
    for line in lines:
        m = _POINT_LINE_RE.match(line)
        if m:
            seen_point = True
            point_texts.append(m.group(2).strip())
            continue
        if not seen_point:
            opener_lines.append(line)
            continue
        stripped = line.strip()
        if not stripped:
            continue
        is_continuation = line[:1].isspace() or stripped.startswith(("-", "*", "•"))
        if is_continuation and point_texts:
            point_texts[-1] = point_texts[-1] + " " + stripped
        else:
            closer_lines.append(line)
    return opener_lines, point_texts, closer_lines


def _rebuild_from_points(opener_lines: List[str], points: List[str], closer_lines: List[str]) -> str:
    """Inverse of _split_numbered_points — rebuild numbered-list text from
    a (possibly filtered) points list. Shared so the rebuild format never
    drifts between the filters that use it.
    """
    opener_part = "\n".join(opener_lines).strip()
    closer_part = "\n".join(closer_lines).strip()
    rebuilt_points = "\n".join(f"{i}. {p}" for i, p in enumerate(points, 1))
    pieces = [p for p in (opener_part, rebuilt_points, closer_part) if p]
    return "\n\n".join(pieces)


def _score_points_against_query(query: str, points: List[str]) -> List[float]:
    """Score each point in *points* for query-relevance via the shared
    cross-encoder reranker, in a single batched .predict() call — reused
    by the contamination trace (log-only) and, once enabled, the Phase-2
    topic-relevance gate. Returns [] on any failure (model not loaded,
    empty points, etc.) rather than raising — a scoring failure must
    never affect the answer itself, only the diagnostic/gate that
    consumes the scores, and those callers already treat an empty list
    as "no signal, don't act."
    """
    if not points or not query:
        return []
    try:
        reranker = _get_shared_reranker()
        pairs = [(query, p) for p in points]
        scores = reranker.predict(pairs)
        return [float(s) for s in scores]
    except Exception as exc:
        logger.debug("[contamination] point-relevance scoring failed: %s", exc)
        return []


def _is_likely_followup(question: str) -> bool:
    """Heuristic: is this question likely a follow-up referencing a previous topic?"""
    words = question.strip().split()
    q_lower_early = question.lower().strip()
    if len(words) > 12:
        # "Long questions are usually self-contained" only holds when the
        # question actually names its own topic. Confirmed live this
        # breaks down otherwise: a 20-word, mostly-filler follow-up ("but
        # i have submitted the money on the monthly so what would happen
        # to my money that i have submitted") named no insurance type at
        # all -- "money" alone is undefined without knowing which policy
        # -- but cleared this gate's old flat 12-word cutoff and got
        # treated as self-contained. retrieval_query then stayed the bare
        # 20-word sentence instead of being anchored to the term-life-
        # policy-expiry context established two turns earlier, and the
        # answer had to hedge across every possible policy type ("if
        # you're referring to term life... if it's a savings/investment-
        # linked policy...") instead of directly answering the one
        # already in play. A genuinely long, standalone new question that
        # names its own topic ("difference between comprehensive and
        # third-party motor insurance for a 5-year-old car") still
        # correctly clears this gate; one that doesn't name any type
        # doesn't, regardless of length. Over-triggering this check for a
        # genuinely fresh long question just costs one extra
        # _reformulate_query call -- that prompt already reliably echoes
        # a self-contained question back unchanged rather than corrupting
        # it, so the failure mode here is asymmetric: worth the latency to
        # avoid the alternative (silently un-anchored retrieval).
        return not _SPECIFIC_TYPE_RE.search(q_lower_early)
    tokens = {w.lower().strip('?.,!') for w in words}
    q_lower = question.lower().strip()

    # Existing checks: pronoun-like tokens and fixed opener phrases
    if bool(tokens & _FOLLOWUP_SIGNALS) or any(q_lower.startswith(op) for op in _FOLLOWUP_OPENERS):
        return True

    # Modifier-only request with no insurance vocabulary → clearly a follow-up.
    # Covers "can you explain with the help of an example" (example, 9 words),
    # "explain in simple language" (simple), "can you explain in more detail" (detail),
    # "in simple language with example", "in detail with simple language", etc.
    # The 6-word keyword check below misses these because of the word-count gate.
    _all_modifier_signals = _DETAIL_SIGNALS | _SIMPLE_SIGNALS
    if (
        any(sig in q_lower for sig in _all_modifier_signals)
        or _wants_example(q_lower)
        or _DETAIL_PATTERN.search(q_lower)
    ):
        from re import compile as _re_compile
        _II_QUICK = _re_compile(
            r"\b(insurance|policy|premium|deductible|coverage|claim|health|medical|"
            r"life|motor|travel|home|vehicle|accident|liability|rider|annuity|pension)\b"
        )
        if not _II_QUICK.search(q_lower):
            return True

    # Additional keyword check: short messages (<= 6 words) that contain a
    # follow-up keyword but NO actual insurance vocabulary are likely referential
    # follow-ups ("explain further", "why though", "expand on that").
    if len(words) <= 6:
        stripped_tokens = {w.strip(".,!?;:()[]{}'\"") for w in words}
        if stripped_tokens & _FOLLOWUP_KEYWORDS:
            # If the question contains real insurance vocabulary it's
            # self-contained, not a follow-up.
            from re import compile as _re_compile
            _INSURANCE_INDICATORS_LOCAL = _re_compile(
                r"\b("
                r"insurance|policy|premium|deductible|coverage|claim|claims|"
                r"insure|insured|insurer|underwriting|underwrite|"
                r"cover|covered|covers|covers? (?:for|against|up to|of|on)|"
                r"premiums|co-pay|copay|deductibles|"
                r"health|medical|hospital|surgery|prescription|medication|"
                r"vehicle|car|motor|auto|bike|two-wheeler|four-wheeler|"
                r"travel|trip|flight|baggage|luggage|cancellation|"
                r"life|term life|whole life|endowment|ulip|"
                r"home|house|property|rental|landlord|tenant|"
                r"accident|disability|critical illness|cancer|"
                r"liability|third.party|comprehensive|"
                r"limit|limits|sum insured|sum.assured|"
                r"maternity|dental|vision|"
                r"agent|broker|renewal|grace period|waiting period|"
                r"no.claim|ncb|bonus|"
                r"nominee|beneficiary|"
                r"claim (?:form|process|settlement|rejection|approval)|"
                r"cashless|reimbursement|"
                r"roadside assistance|towing|"
                r"personal accident|"
                r"retirement|pension|annuity|"
                r"finance|financial|investment|savings|"
                r"hdfc ergo|icici|bajaj|tata aig|reliance|"
                r"new india|oriental|national|united india|"
                r"irda|regulator|"
                r"cover note|certi.* of insurance|"
                r"aog|marine|cargo|"
                r"group insurance|corporate|"
                r"rider|add.on"
                r")\b"
            )
            if not _INSURANCE_INDICATORS_LOCAL.search(q_lower):
                return True

    # A short question using only GENERIC insurance-process vocabulary
    # ("claim", "premium", "documents", "cost"...) but no SPECIFIC
    # insurance-type noun (motor, life, health, travel, home...) still
    # depends on whatever type was being discussed — it isn't self-contained
    # just because it contains an insurance word. Confirmed live: "How do I
    # file a claim" asked right after "What is life insurance" fell through
    # every check above (no follow-up opener, no _FOLLOWUP_KEYWORDS match,
    # no detail/simple/example modifier) and was treated as a fresh,
    # standalone question — so it skipped _reformulate_query entirely and
    # retrieval ran on the bare, topic-less phrase, returning confident
    # motor-insurance claim content (roadside assistance, police report)
    # instead of anything about life insurance, or a refusal.
    if len(words) <= 10:
        _GENERIC_PROCESS_RE = re.compile(
            r"\b(claim|claims|premium|premiums|deductible|deductibles|"
            r"policy|policies|documents?|paperwork|process|renew|renewal|"
            r"cancel|cancellation|cost|price|apply|application)\b"
        )
        if _GENERIC_PROCESS_RE.search(q_lower) and not _SPECIFIC_TYPE_RE.search(q_lower):
            return True

    return False


_REACTION_LEADIN_RE = re.compile(
    r"^(oh\s+)?(okay|ok|alright|got it|i see|gotcha|ah|"
    r"that makes sense|makes sense|interesting|nice|cool|great|awesome|"
    r"good to know|noted|thanks|thank you)\b",
    re.IGNORECASE,
)
_REACTION_QUESTION_RE = re.compile(
    r"\?|\b(what|how|why|when|where|which|who|can you|could you|do you|does it|"
    r"is it|is there|are there|will it|would it|should i)\b",
    re.IGNORECASE,
)


def _is_conversational_reaction(question: str) -> bool:
    """True for a longer acknowledgment/reaction with no retrieval-worthy
    question content, e.g. "Oh okay that makes sense, I didn't know it
    needed forced entry specifically." Confirmed live: at 13 words this is
    past _is_likely_followup's 12-word "long questions are usually
    self-contained" cutoff, so it fell through to standalone KB retrieval,
    found nothing (it isn't a question, there's nothing to retrieve), and
    triggered an unnecessary refusal + human-escalation email for a
    message that was never actually asking anything.

    Deliberately a separate check rather than folded into
    _is_likely_followup — that function's word-count gate is tuned for
    actual follow-up QUESTIONS, and loosening it risks the same
    fragile-signal-list regressions documented elsewhere in this file
    (_SIMPLE_SIGNALS/_DETAIL_SIGNALS). This only needs to catch pure
    acknowledgments: it requires a reaction lead-in AND the absence of any
    question-request pattern, so a message like "oh interesting, but does
    it also cover theft without forced entry?" still gets treated as a
    real question, not a reaction, and goes through normal retrieval.
    """
    q = question.strip()
    if not q or not _REACTION_LEADIN_RE.match(q):
        return False
    return not _REACTION_QUESTION_RE.search(q)


# Specific insurance-type nouns, used only to deterministically re-anchor a
# reformulated follow-up whose LLM rewrite silently dropped the topic — see
# the repair step inside _reformulate_query for why this exists.
#
# This list is fixed-phrase and therefore fragile by construction (same
# class as _SIMPLE_SIGNALS/_DETAIL_SIGNALS — see the fragile-signal-lists
# note elsewhere in this file): any KB topic missing from it is invisible to
# both the "does the question already name a type" check AND the "what type
# did history establish" check below. Confirmed live: "actually never mind
# that, what does aviation insurance cover" (a self-contained, explicitly
# topic-switched question) failed BOTH checks because "aviation" wasn't
# here, so the repair below wrongly force-appended "medical insurance" left
# over from the prior turn — corrupting retrieval into pulling travel-
# insurance chunks and answering with travel-insurance content (medical
# evacuation, air ambulance) under the "aviation insurance" heading. Cross-
# checked the KB's actual document text for other real, well-represented
# (7+ chunk) types missing here — aviation, burglary, workmen's
# compensation, fidelity guarantee, engineering, business interruption,
# micro-insurance, and unit-linked all had solid, dedicated content and
# were missing; added all of them rather than just the one that happened to
# get hit, per the standing instruction to fix the whole bug class.
_ANCHOR_TYPE_RE = re.compile(
    r"\b(term|motor|car|vehicle|auto|bike|two.wheeler|four.wheeler|"
    r"life|whole\s*life|endowment|ulip|unit.linked|"
    r"health|medical|"
    r"travel|trip|aviation|"
    r"home|house|property|householder|"
    r"marine|cargo|fire|burglary|"
    r"liability|third.party|public\s*liability|"
    r"critical\s*illness|"
    r"group|corporate|micro|"
    r"personal\s*accident|personal\s*insurance|road\s*accident|accident|disability|workmen.?s?\s*compensation|"
    r"employer.?s?\s*liability|fidelity(?:\s*guarantee)?|"
    r"engineering|business\s*interruption|"
    r"retirement|pension|annuity|takaful|"
    r"crop|agricultur\w*)\b(\s+insurance)?",
    re.IGNORECASE,
)

# The hardcoded list above will keep having this exact gap for every future
# insurance type a new KB document introduces — it was already missed once
# for 9 real, well-represented types before anyone noticed. Rather than
# relying solely on a human to notice and patch the list again next time,
# derive additional candidate types directly from the KB's own document
# text at runtime, so a newly-added document's type is picked up
# automatically without a code change. This is intentionally additive
# (union with the hardcoded list, never a replacement) — the hardcoded list
# is a reliable floor; the mined list only ever adds coverage on top of it,
# so a bad or missing mined term can't remove something that used to work.
_STOPWORD_TAIL_RE = re.compile(
    r"^(the|a|an|in|on|at|for|to|and|or|with|by|from|as|is|are|was|were|"
    r"this|that|these|those|notes?|project|safe|duty|double|effective|"
    r"comprehensive|differentiation|construction|energy|other|all|each|"
    r"such|said|above|below|certain|various|special|select|digital|"
    r"modified|pure|floater|united|india|royal|london|globe|triton|"
    r"oriental|indian|mercantile|national|public|sector|commercial|"
    r"operational|variable|savings|regulations?|affecting|framework|"
    r"governance|licensing|provisions|categor(?:y|ies)|role|history|"
    r"report|period|preconditions|commencement|registration|"
    r"certification|coverage|course|application|countering|fraud|agents|"
    r"appointed|controller|fellow|wing)$"
)
_KB_TYPE_PHRASE_RE = re.compile(r"\b((?:[A-Z][a-z]+'?s?\s+){0,2}[A-Z][a-z]+)\s+Insurance\b")
_MIN_KB_TYPE_FREQ = 3

def _discover_kb_anchor_types(vector_store) -> set:
    """Mine specific insurance-type phrases directly from the KB's chunk
    text, to extend _ANCHOR_TYPE_RE with types no one has hardcoded yet.

    Looks for 1-3 capitalized words immediately before "Insurance" (e.g.
    "Aviation Insurance", "Officers Liability Insurance") — requiring
    capitalization is what keeps this clean without heavy NLP: genuine
    product/type names are capitalized in this KB's prose, while stray
    connector words ("the", "of the", "an") almost never are. What survives
    that filter still needs two more passes to be usable as a topic anchor:
    a small blocklist for document-structure/regulatory vocabulary and
    insurer company names that happen to be capitalized too ("Oriental
    Insurance", "United India Insurance" are companies, not types), and a
    minimum-frequency bar to drop one-off mentions. A last pass drops any
    bare single word that's just a fragment of an already-accepted
    multi-word phrase (e.g. bare "linked" once "unit linked" is present) —
    single words carry the most false-positive risk since they're most
    likely to also be ordinary English words.

    Also unions in every value of the KB's own `policy_type` chunk-metadata
    field (e.g. "cyber", "marine") with NO filtering — confirmed live this
    was a real gap the prose-mining alone missed: "cyber insurance" only
    appears in 2 chunks in this KB, below the 3-chunk frequency bar, so it
    silently failed the same topic-switch bug the rest of this mechanism
    exists to prevent ("switching to cyber insurance" after a marine-
    insurance turn got "marine insurance" wrongly re-appended). `policy_type`
    is structured, curated data assigned at ingestion time, not prose
    mined from free text, so it carries none of the false-positive risk
    (company names, sentence fragments) the text-mining pass has to guard
    against — safe to trust unconditionally, and it catches genuinely
    under-represented types the frequency bar would otherwise miss.
    """
    counts: dict = {}
    try:
        docs = vector_store.get_all_by_filter({})
    except Exception:
        logger.debug("[_discover_kb_anchor_types] KB scan failed, using hardcoded list only", exc_info=True)
        docs = []
    for doc in docs:
        text = getattr(doc, "page_content", "") or ""
        for m in _KB_TYPE_PHRASE_RE.finditer(text):
            phrase = m.group(1).strip().lower()
            if len(phrase) < 3:
                continue
            if any(_STOPWORD_TAIL_RE.match(w) for w in phrase.split()):
                continue
            counts[phrase] = counts.get(phrase, 0) + 1
    clean = {p for p, c in counts.items() if c >= _MIN_KB_TYPE_FREQ}
    multiword = {p for p in clean if " " in p}
    final = set()
    for p in clean:
        if " " not in p and any(p in mw.split() for mw in multiword):
            continue
        final.add(p)

    try:
        for policy_type in vector_store.list_values("policy_type"):
            if policy_type:
                final.add(policy_type.strip().lower())
    except Exception:
        logger.debug("[_discover_kb_anchor_types] policy_type lookup failed", exc_info=True)

    return final


_dynamic_anchor_pattern_cache: Optional[re.Pattern] = None

def _get_dynamic_anchor_pattern(vector_store) -> re.Pattern:
    """_ANCHOR_TYPE_RE extended with types mined live from the KB, computed
    once per process and cached — the KB doesn't change size often enough
    to justify re-scanning every request. Call _reset_dynamic_anchor_cache()
    after adding new documents to pick up their types without a restart.
    Falls back to the bare hardcoded pattern on any failure so a KB-scan
    problem degrades to the old (safe, previously-shipped) behavior rather
    than breaking reformulation entirely.
    """
    global _dynamic_anchor_pattern_cache
    if _dynamic_anchor_pattern_cache is not None:
        return _dynamic_anchor_pattern_cache
    try:
        discovered = _discover_kb_anchor_types(vector_store)
        new_terms = sorted(discovered - _hardcoded_anchor_terms())
        if new_terms:
            logger.info("[_get_dynamic_anchor_pattern] mined %d new anchor type(s) from KB: %r", len(new_terms), new_terms)
        if new_terms:
            extra = "|".join(re.escape(t) for t in new_terms)
            pattern = re.compile(
                _ANCHOR_TYPE_RE.pattern.replace(r")\b(\s+insurance)?", f"|{extra})\\b(\\s+insurance)?"),
                re.IGNORECASE,
            )
        else:
            pattern = _ANCHOR_TYPE_RE
        _dynamic_anchor_pattern_cache = pattern
        return pattern
    except Exception:
        logger.debug("[_get_dynamic_anchor_pattern] falling back to hardcoded pattern", exc_info=True)
        return _ANCHOR_TYPE_RE


# Plain-text mirror of the alternatives inside _ANCHOR_TYPE_RE, used only to
# avoid logging an already-hardcoded term as "newly discovered" from the KB.
# Kept as an explicit set rather than reverse-parsed from the compiled
# pattern's regex source — an earlier version tried stripping backslashes
# out of the pattern string directly, which mangled `\s*`-joined terms
# ("personal\s*accident" -> "personalsaccident" instead of "personal
# accident") and made genuinely-already-hardcoded terms look new in the
# discovery log. This set going stale (someone edits _ANCHOR_TYPE_RE
# without updating it) only costs a cosmetic duplicate log line, never
# breaks matching — the dynamic pattern still unions in the "duplicate"
# term harmlessly.
_HARDCODED_ANCHOR_TERMS = {
    "term", "motor", "car", "vehicle", "auto", "bike", "two wheeler", "four wheeler",
    "life", "whole life", "endowment", "ulip", "unit linked",
    "health", "medical",
    "travel", "trip", "aviation",
    "home", "house", "property", "householder",
    "marine", "cargo", "fire", "burglary",
    "liability", "third party", "public liability",
    "critical illness",
    "group", "corporate", "micro",
    "personal accident", "personal insurance", "disability", "workmens compensation",
    "employers liability", "fidelity", "fidelity guarantee",
    "engineering", "business interruption",
    "retirement", "pension", "annuity", "takaful",
    "crop", "agriculture",
}

def _hardcoded_anchor_terms() -> set:
    return _HARDCODED_ANCHOR_TERMS


def _reset_dynamic_anchor_cache() -> None:
    """Force the next _get_dynamic_anchor_pattern() call to re-scan the KB —
    call this after documents are added/removed so new types are picked up
    without a container restart."""
    global _dynamic_anchor_pattern_cache
    _dynamic_anchor_pattern_cache = None


def _last_anchor_type_match(text: str, pattern: re.Pattern = _ANCHOR_TYPE_RE) -> Optional[str]:
    """Last (most recent) specific-insurance-type phrase mentioned in *text*,
    or None.

    Checks the most recent "User:" turn first — the user's own wording is
    the authoritative signal for what topic they actually asked about. The
    "Assistant:" turn often re-describes it via a broader category ("Term
    insurance is a type of life insurance...") which, under a plain
    last-match-anywhere search, would wrongly outrank the real, more
    specific topic the user named ("term") just because "life insurance"
    happens to appear later in the same turn's prose. Only falls back to
    searching the whole text when the user's own turn has no type mention
    (e.g. a topic introduced solely by the assistant, never restated).

    The fallback (assistant-only) search requires the matched type to
    appear at least twice in *text* — a single occurrence is often just an
    illustrative aside within an otherwise generic answer, not the actual
    established topic. Confirmed live: "What does the policy document
    include?" (names no type) got an answer that mentioned "Fire Insurance"
    exactly once, as a passing example among several ("...in Fire Insurance
    the particulars of the building...", alongside a car-insurance example
    in the same sentence) — a plain last-match-anywhere search still picked
    it up as "the established topic" and the repair step below injected
    "for fire insurance" into a follow-up about something else entirely
    ("how can relatives be informed about the policy?"), reintroducing the
    exact type-hallucination failure the _reformulate_query LLM-prompt
    guardrail was built to prevent — this deterministic repair runs after
    that LLM call and isn't covered by its prompt instructions at all. A
    genuinely-established topic (the documented term-insurance/"how to
    claim it?" case this repair exists for) is discussed substantively
    enough to be named more than once, so the frequency check doesn't
    weaken the intended case.
    """
    user_turns = re.findall(r"User:\s*(.*?)(?=\n(?:User|Assistant):|\Z)", text, re.DOTALL)
    if user_turns:
        user_matches = list(pattern.finditer(user_turns[-1]))
        if user_matches:
            phrase = user_matches[-1].group(0).strip()
            if not re.search(r"insurance\s*$", phrase, re.IGNORECASE):
                phrase = f"{phrase} insurance"
            return phrase

    matches = list(pattern.finditer(text))
    if not matches:
        return None
    base_counts: dict = {}
    for m in matches:
        base = m.group(0).strip().lower().split()[0]
        base_counts[base] = base_counts.get(base, 0) + 1
    phrase = matches[-1].group(0).strip()
    if base_counts[phrase.lower().split()[0]] < 2:
        return None
    if not re.search(r"insurance\s*$", phrase, re.IGNORECASE):
        phrase = f"{phrase} insurance"
    return phrase


# Legislative/tax/administrative boilerplate — a chunk matching this is
# almost never what a consumer-facing question is actually asking about,
# even when it happens to name the right insurance type (e.g. "the Act
# guarantees amounts assured by LIC policies" mentions "life insurance"
# but is about a government solvency guarantee, not payout amounts). See
# _prioritize_topic_chunks's docstring for the concrete failure this fixes.
#
# "ICP \d+" and "the supervisor" extend this to insurance-REGULATOR/
# supervisory-framework boilerplate (IAIS Insurance Core Principles text —
# licensing, board-member suitability, governance requirements aimed at
# insurers-as-regulated-entities). Confirmed live: "How can relatives be
# informed about the policy?" cross-encoder-scored a chunk about board-
# member/significant-owner suitability licensing at 0.94 — higher than
# every genuinely relevant chunk in the pool — even though the selected
# rerank window didn't contain the word "relative" at all. Same underlying
# judgment as the original regulatory-boilerplate case: this class of
# supervisor-facing governance text is essentially never what a consumer
# question wants, regardless of how the cross-encoder happens to score it.
# "the supervisor" specifically (not "the regulator"/"the authority" more
# broadly, which are too generic and risk false-positiving on unrelated
# content) because in this KB's consumer-facing prose the acting party is
# always "the insurer"/"the insured"/"the policyholder"/"the agent" — "the
# supervisor" only appears in the regulatory-oversight framework sections.
# "grievance redressal|adjudicat\w*" extends this to the same category
# confirmed live via the hollow-answer investigation: "my claim got
# rejected, what do I do" retrieved a chunk about IRDA's statutory power to
# adjudicate disputes between insurers and intermediaries — regulatory/
# supervisory content about the regulator's OWN powers, not a consumer-
# facing "how do I appeal my own claim" answer, but lexically close enough
# ("claim", "dispute") to score highly on the cross-encoder anyway. Scoped
# tightly to just these two terms after confirming live they're safe:
# every occurrence of "adjudicat*" in this KB (8/8) and a sample of
# "grievance redressal" occurrences are exclusively inside the IRDA
# statutory-powers listing, never inside consumer-facing policy content.
# Broader candidates considered and REJECTED after they caused a real
# regression on a well-covered compound question (fire + burglary
# insurance): bare "irda\b" and "insurance intermediar*" both appear
# routinely in ordinary consumer-education content in this KB (a table of
# contents entry, a "what is an intermediary" explainer chunk) — nowhere
# near narrow enough to safely exclude without also demoting genuinely
# relevant chunks that happen to mention the regulator or agents/brokers
# in passing.
_REGULATORY_BOILERPLATE_RE = re.compile(
    r"\b(the act|the bill|stamp duty|income tax act|central government|"
    r"section\s+\d+[a-z]?\b.{0,20}\bact\b|"
    r"guarantee[sd]?\s+by\s+the\s+(central\s+)?government|"
    r"amends?\b|gross\s+total\s+income|"
    r"icp\s*\d+|the\s+supervisor|"
    r"grievance\s+redressal|adjudicat\w*)\b",
    re.IGNORECASE,
)

# Chapter-end review/index material — never substantive answer content, but
# can still slip past the topic-match check above because it name-drops
# every type covered in that chapter as part of a summary or self-test
# question ("5. What do you mean by Motor Insurance?"). Confirmed live:
# "Explain motor insurance in detail" top-scored (0.475, the single
# highest-ranked candidate) a chunk that is actually a "General Insurance
# Products" survey listing crop and fidelity insurance, plus its chapter's
# self-test questions — it only survived the topic filter because "Motor
# Insurance" happens to appear inside one of those review questions, not
# because the chunk is actually about motor insurance. A chunk whose real
# subject is a review/index of MULTIPLE topics should never outrank one
# that's actually about the queried topic, regardless of what it name-drops.
_CHAPTER_REVIEW_RE = re.compile(
    r"\b(self.test questions?|lesson round.?up|(?:are\s+)?meant\s+for\s+re.capitulation)\b",
    re.IGNORECASE,
)


def _prioritize_topic_chunks(retrieval_query: str, chunks: list) -> list:
    """Reorder *chunks* so ones whose content mentions the query's named
    insurance type (e.g. "health insurance") come before ones that don't —
    same relative order preserved within each group (stable sort), so this
    only ever changes ORDER, never which chunks are included.

    Exists for two related failure modes, both confirmed live:
    (a) a generic, universal-to-any-policy glossary chunk outranking
    genuinely topic-specific content sitting right next to it in the same
    retrieved pool (see DETAILED_GROUNDED_PROMPT/STRICT_GROUNDED_PROMPT's
    "prioritize topic-specific content" rule, which helped but wasn't
    reliable alone), and (b) a chunk from a DIFFERENT insurance type
    entirely outranking the correct one because the wording happens to be
    lexically/semantically similar — confirmed with "are all types of
    illness covered under health insurance?": the top-scoring chunk (0.78)
    was travel insurance's exclusion list ("illnesses or injuries that
    occurred... before the start of the journey"), correctly rejected by
    the semantic grounding check as not actually about health insurance —
    but that refusal meant a REAL health-insurance exclusion chunk sitting
    lower in the same pool (0.10, cosmetic/aesthetic treatment exclusions)
    never got a chance to ground the answer either. Reordering so the
    correct-topic chunk is physically first gives it a chance to be what
    the coverage/grounding checks and generation actually see and use.

    A no-op when the query doesn't name a specific type (single-word
    concept lookups like "what is a deductible" are unaffected).

    Chunks matching _REGULATORY_BOILERPLATE_RE never count as a "topic
    match" for promotion, even when they literally contain the topic
    phrase. Confirmed live: "How much does life insurance pay out?"
    promoted a weak (0.10) chunk about the Insurance Act's central-
    government solvency guarantee for LIC purely because it contains the
    words "life insurance" — ahead of a higher-scoring but wrong-topic
    (health insurance) chunk. That regulatory chunk doesn't answer "how
    much", but the grounding check judged it "relevant enough" anyway
    (reproduced 15/15 across 5 different prompt phrasings, including ones
    naming this exact failure as an explicit exclusion — this is a hard
    model limitation, not a prompt-wording gap, so it's addressed here at
    the reordering step instead of asking the LLM to see through it).
    Legislative/tax/administrative boilerplate is essentially never what a
    consumer-facing "what/how much/how" question is looking for even when
    it happens to name the right insurance type — see _reformulate_query's
    existing "do not add a specific regulation, act, section... unless
    already named" rule for the same underlying judgment applied earlier
    in the pipeline.

    The boilerplate deprioritization runs even when the query names NO
    specific insurance type (topic is None below) — confirmed live with
    "How can relatives be informed about the policy?" (no type mentioned
    anywhere in the conversation): a supervisory-licensing boilerplate
    chunk outranked every genuinely relevant chunk by cross-encoder score
    alone, and topic-matching had nothing to gate on since there was no
    named topic to check against. Boilerplate is a category judgment
    independent of topic, so it applies unconditionally; only the topic
    promotion half of this function needs a named type to compare against.
    """
    m = _ANCHOR_TYPE_RE.search(retrieval_query)
    topic = m.group(0).lower() if m else None
    def _rank(c) -> int:
        content = (getattr(c, "page_content", "") or "").lower()
        if _REGULATORY_BOILERPLATE_RE.search(content) or _CHAPTER_REVIEW_RE.search(content):
            return 1
        if topic and topic not in content:
            return 1
        return 0
    return sorted(chunks, key=_rank)


async def _reformulate_query(question: str, history: str, anchor_pattern: re.Pattern = _ANCHOR_TYPE_RE) -> Optional[str]:
    """Rewrite a follow-up question as a standalone, natural-language question
    using conversation history — used both for retrieval and, for detected
    follow-ups, as the literal question text shown to the generation prompt.

    Examples:
      history: "User: tell me about life insurance\nLayla: Life insurance ..."
      question: "what about premiums?"
      returns: "What is the premium amount for life insurance?"

    Uses max_tokens=30 and a 4-second timeout so it adds <0.5 s to latency.

    Returns None on genuine failure (backend call failed/timed out, or
    returned degenerate output) — the caller then falls back to
    _reformulate_with_history. Returns *question* UNCHANGED (not None) when
    the model correctly determines no rewrite is needed — this is a
    successful outcome, not a failure, and must not trigger that fallback:
    the prompt below now explicitly instructs the model to echo an
    already-self-contained follow-up back verbatim rather than force a
    paraphrase (confirmed live: unnecessary rewrites of a fine question —
    adding "typically", "regarding the details of" — introduced wording
    that swung the downstream cross-encoder's relevance score by 10-30x for
    no benefit). Collapsing "unchanged because unnecessary" into the same
    signal as "failed" would defeat that fix by re-triggering the history-
    snippet fallback for a question that didn't need any rewrite at all.
    """
    # Use only the last 2 turns (the single most recent Q&A pair) of history.
    # Used to slice by raw newline count instead of turn boundaries —
    # confirmed live: a multi-paragraph, numbered-list detailed answer alone
    # spans more than 6 newlines, so "last 6 lines" only captured the tail
    # end of the MOST RECENT answer (e.g. "...the lump-sum payment helps
    # Sarah focus on recovery") and never saw the subject established earlier
    # in that same answer ("critical illness insurance"). The follow-up "what
    # is its purpose" got reformulated to "lump-sum payment purpose in
    # insurance policy" — grammatically fine, but missing the actual topic —
    # so retrieval pulled unrelated content and the question was refused.
    # _split_history_turns() slices on "User:"/"Assistant:" boundaries so a
    # long answer is kept whole instead of being cut mid-turn.
    #
    # Window then narrowed 6 -> 2: confirmed live again with a longer, multi-
    # topic conversation (marine insurance -> takaful -> an off-topic aside ->
    # term insurance) — with 3 Q&A pairs of history in view, "explain in
    # detail with an example" got reformulated to "takaful insurance
    # principle example real case", pulling the more distinctive-sounding
    # topic from two turns back instead of the actually-current one (term
    # insurance) from the immediately preceding turn. Generic modifier-only
    # follow-ups ("give me an example", "explain in detail") are asking about
    # whatever was JUST discussed — the single most recent turn is both
    # necessary and sufficient, and including more only risks the model
    # anchoring on an earlier, more salient-sounding topic instead. This
    # matches _contextualize_query's existing narrower window for the same
    # reason.
    recent = '\n'.join(_split_history_turns(history)[-2:])
    # Output a complete, natural-language QUESTION — not a terse keyword
    # string. The old prompt asked for compact "textbook vocabulary" phrases
    # (e.g. "term insurance detailed explanation example"), which is exactly
    # what the few-shot examples below used to demonstrate. Confirmed live:
    # that exact keyword-salad phrase, tried directly against retrieval,
    # returned an EMPTY context (no chunks matched well enough) even though
    # the natural phrasing of the same request — "explain term insurance in
    # detail with an example" — retrieved rich, correct content and answered
    # fully. The embedding model matches natural sentences against this KB's
    # prose far better than a stripped keyword list. This same string also
    # doubles as the literal "question" shown to the generation prompt for
    # follow-ups (see prompt_question below), where a keyword salad reads as
    # not-a-real-question and biases the model toward refusing — a complete
    # question fixes both problems at once.
    prompt = f"""Rewrite the follow-up question as a complete, standalone, natural-language question using the conversation context.
Use precise insurance/legal terms that would appear in a textbook, but phrase it as a real question a person would ask — not a keyword list.
If the follow-up is ALREADY a complete, self-contained question that doesn't
depend on anything in the conversation below to be understood — no pronoun,
no missing topic, nothing implicit — output it EXACTLY AS-IS, character for
character, changing nothing. Do not add hedging words ("typically",
"usually"), do not add a topic that isn't needed, do not paraphrase just to
sound more natural — an unnecessary rewrite is not an improvement, and
different wording of an already-fine question can retrieve worse results
than the original. Only rewrite when something genuinely needs resolving.
The conversation below is ONLY the single most recent exchange — always ground
the follow-up in that topic, never in anything outside what's shown here.
Do NOT add a specific regulation, act, section, jurisdiction, or authority
(e.g. "under federal law", "under section 80C", "under IRDA regulations")
unless the conversation itself already named it — inventing one, even a
plausible-sounding one, measurably hurts retrieval when the actual source
material frames it differently or doesn't cite a specific provision at all.
Same rule for the insurance TYPE itself: do NOT introduce a specific
product name (e.g. "fire insurance", "marine insurance") that appears
NOWHERE in the conversation above AND nowhere in the follow-up itself —
only reuse a product name that was already used somewhere. This is about
not INVENTING a new product name out of nothing — it does not mean
dropping words that already appear in the follow-up; always keep
everything the follow-up itself asks about (e.g. "how can relatives be
informed" must still mention relatives in your rewrite).
If the follow-up ALREADY asks about multiple things joined by "and" (e.g.
"what is X and how to Y"), your rewritten question must keep EVERY part —
never drop a part just because it resembles something already discussed.
Output ONLY the rewritten question — no quotes, no explanation, nothing else.

Examples:
  Context: "User: what does the policy document include?\nLayla: The policy document includes the name and address of the insured, sum insured, period of insurance..."
  Follow-up: "How can relatives be informed about the policy?" → "How can relatives be informed about the policy?" (already self-contained — output unchanged)

  Context: "User: tell me about life insurance\nLayla: Life insurance pays out..."
  Follow-up: "what about premiums?" → "What is the premium amount for life insurance?"

  Context: "User: explain life insurance\nLayla: Life insurance protects your family..."
  Follow-up: "is it tax deductible?" → "Are life insurance premiums tax deductible?"

  Context: "User: explain reinsurance\nLayla: Reinsurance is when insurers share risk..."
  Follow-up: "is it legally required?" → "Is reinsurance a legal requirement?"

  Context: "User: what is term insurance\nLayla: Term insurance is a type of life insurance..."
  Follow-up: "what is term insurance and how to claim it?" → "What is term insurance and how do you claim a term insurance policy?"

  Context: "User: what is subrogation\nLayla: Subrogation means the insurer steps in..."
  Follow-up: "give me an example" → "Can you give a real-life example of subrogation?"

  Context: "User: what is a deductible\nLayla: A deductible is what you pay first..."
  Follow-up: "how is it calculated?" → "How is a deductible amount calculated?"

Conversation:
{recent}

Follow-up: {question}
Search query:"""
    reformulated = await _backend_completion(prompt, max_tokens=30, timeout=4)
    if reformulated:
        reformulated = reformulated.strip().strip('"\'')
        if len(reformulated) >= 3:
            # Deterministic repair for a confirmed non-determinism gap: even
            # at temperature=0, this call has been observed live to drop the
            # topic anchor on the *same* input across repeated calls — e.g.
            # "How to claim it?" after a term-insurance turn reformulated to
            # the correctly-anchored "How is a claim for term insurance
            # processed?" on one call, and the topic-less "What are the steps
            # involved in submitting a claim?" moments later on an identical
            # retry (same question, same history). A topic-less reformulation
            # isn't a refusal — it's a *different, legitimately generic*
            # question that goes on to confidently ground against whatever
            # generic claims content ranks highest, a wrong-topic answer that
            # nothing downstream can distinguish from an intentionally
            # generic question. Only repair when the topic could ONLY have
            # come from history (the raw follow-up itself names no specific
            # type) and the rewrite dropped the type history established.
            # Deterministic repair for the inverse gap, also confirmed live
            # (reproduced 3x, intermittently): the rewrite sometimes INVENTS
            # a specific insurance type out of nothing — "How can relatives
            # be informed about the policy?" after a generic policy-document
            # turn (no type named anywhere in the conversation) reformulated
            # to "...policy documents for fire insurance?", so retrieval
            # pulled fire-insurance chunks, the semantic grounding check
            # correctly rejected them, and the question was refused. The
            # prompt above already forbids exactly this and the model still
            # does it intermittently — same model limitation as the anchor-
            # drop case below, so same remedy: enforce in code. Only phrases
            # with an explicit "insurance" suffix are stripped; a bare type
            # word ("property", "group", "fire") is too ambiguous to remove
            # safely from an otherwise-good rewrite. Runs BEFORE the anchor-
            # append repair so that when history DOES name a real type, a
            # stripped wrong-type rewrite can still be re-anchored to it.
            _known_topics = f"{question}\n{recent}"
            # Hyphen/space forms are treated as interchangeable when checking
            # whether a type is "already known" — confirmed live: the raw
            # follow-up said "third party insurance" (space), the LLM's own
            # rewrite correctly wrote "third-party insurance" (hyphen), and
            # a literal-text comparison between the two treated the hyphen
            # form as a brand-new invented type never seen anywhere,
            # stripping "third-party insurance" out of an otherwise-correct
            # rewrite — "How is liability insurance different from
            # third-party insurance?" became the badly broken "How is
            # liability insurance different from?", which then answered a
            # completely different (and unasked) comparison against property
            # insurance instead. Splitting the matched phrase into words and
            # rejoining with a hyphen-or-space-tolerant separator fixes the
            # comparison without weakening the check itself — it's still an
            # exact word-sequence match, just insensitive to which of the
            # two equally-common spellings either side happened to use.
            def _flexible_known(phrase: str) -> bool:
                _words = [w for w in re.split(r"[\s-]+", phrase) if w]
                if not _words:
                    return False
                _pattern = r"[\s-]+".join(re.escape(w) for w in _words)
                return bool(re.search(rf"\b{_pattern}\b", _known_topics, re.IGNORECASE))

            for _ in range(3):  # rescan after each strip; bounded
                _invented = next(
                    (
                        _m for _m in anchor_pattern.finditer(reformulated)
                        if _m.group(2)
                        and not _flexible_known(_m.group(1))
                    ),
                    None,
                )
                if _invented is None:
                    break
                _stripped = re.sub(
                    rf"(?:\s+(?:for|of|in|under|on|about|regarding|with))?\s*\b{re.escape(_invented.group(0))}\b",
                    "",
                    reformulated,
                    flags=re.IGNORECASE,
                )
                _stripped = re.sub(r"\s{2,}", " ", _stripped).strip(" ,;:")
                if _invented.start() == 0 or len(re.findall(r"\w+", _stripped)) < 3:
                    # Invented type was the sentence subject, or stripping
                    # gutted the rewrite — the raw follow-up is safer than
                    # either a wrong-topic or a broken query.
                    logger.info(
                        "[REFORM] invented-type repair: %r names %r found nowhere in the conversation — falling back to raw follow-up",
                        reformulated, _invented.group(0),
                    )
                    reformulated = question
                    break
                logger.info(
                    "[REFORM] invented-type repair: %r names %r found nowhere in the conversation — stripped to %r",
                    reformulated, _invented.group(0), _stripped,
                )
                reformulated = _stripped
            _hist_anchor = _last_anchor_type_match(recent, pattern=anchor_pattern)
            if (
                _hist_anchor
                and not anchor_pattern.search(question)
                and not anchor_pattern.search(reformulated)
            ):
                logger.info(
                    "[REFORM] topic-anchor repair: %r missing %r from history — appending",
                    reformulated, _hist_anchor,
                )
                reformulated = f"{reformulated.rstrip('?.! ')} for {_hist_anchor}?"
            logger.info("[REFORM] %r -> %r", question, reformulated)
            return reformulated
    return None


_INTENT_PROMPT = """\
Read the insurance question below and classify its content words/phrases into two groups.

SPECIFIC = names a particular concept, country, provider, rule, or number that only
           documents actually discussing that exact thing would contain.
GENERIC  = so common it appears in almost EVERY insurance document regardless of
           topic — quality/selection words (best, choose, recommend), audience
           words (family, individual, personal), or filler (important, options).

Output EXACTLY two lines, nothing else. Use "(none)" if a group is empty.
Keep specific compound terms together (e.g. "no-claim bonus", "free look period").

Examples:
Question: "what is a no-claim bonus"
Specific: no-claim bonus
Generic: (none)

Question: "what is a deductible"
Specific: deductible
Generic: (none)

Question: "choosing the best insurance policy in the uae for families"
Specific: uae
Generic: choosing, best, insurance, policy, families

Question: "how are premiums calculated for fire insurance"
Specific: fire insurance, premium
Generic: (none)

Question: "what should i know about maternity coverage for individuals vs families"
Specific: maternity
Generic: should, know, coverage, individuals, families

Question: {question}
"""


async def _extract_intent_topics(question: str) -> set[str]:
    """Fast LLM call (runs in parallel with retrieval) to extract core,
    discriminating topic words from the question for the coverage check.

    max_tokens=60 so the round-trip is <1 s on an idle backend.
    Falls back to an empty set on any error — caller uses regex fallback.
    """
    try:
        prompt = _INTENT_PROMPT.format(question=question)
        raw = await _backend_completion(prompt, max_tokens=60, timeout=5)
        if not raw:
            return set()

        specific_line = ""
        generic_line = ""
        for line in raw.split("\n"):
            line_l = line.strip().lower()
            if line_l.startswith("specific:"):
                specific_line = line.split(":", 1)[1] if ":" in line else ""
            elif line_l.startswith("generic:"):
                generic_line = line.split(":", 1)[1] if ":" in line else ""

        # Parse phrases like "no-claim bonus, deductible" keeping
        # multi-word/hyphenated compounds intact for phrase matching.
        topics: set[str] = set()
        for phrase in specific_line.lower().split(","):
            phrase = re.sub(r"[^a-z\- ]", "", phrase).strip()
            if phrase and phrase not in ("none", "n/a"):
                topics.add(phrase)

        # "Generic:" is still requested in the prompt — contrasting it
        # against "Specific:" helps the model reason about the split —
        # but its output is no longer persisted anywhere. An earlier
        # version fed it into a disk-backed learned-stopwords set
        # shared across queries, which caused two confirmed bugs: (1)
        # off-topic test/user queries taught it nonsense words with
        # nothing to do with insurance, and (2) each deployment
        # accumulates its own independent, un-synced state, so
        # identical code + identical KB behaved differently across
        # environments depending on each one's unrelated query
        # history. Kept for debug visibility only.
        generic_words: set[str] = set()
        for phrase in generic_line.lower().split(","):
            phrase = re.sub(r"[^a-z\- ]", "", phrase).strip()
            if phrase and phrase not in ("none", "n/a"):
                generic_words.update(w for w in phrase.split() if len(w) >= 3)

        logger.debug("[INTENT] %r → specific=%s generic=%s", question, topics, generic_words)
        return topics
    except Exception as exc:
        logger.debug("[INTENT] extraction failed (%s) — using regex fallback", exc)
        return set()


async def _classify_query_policy_type_llm(query: str) -> str:
    """LLM fallback for query policy_type, used only when the free regex
    pass (classify_query_policy_type) can't confidently name one.

    Most real queries don't name their type in the exact textbook phrase
    regex looks for — confirmed live: "what's not covered if my car is
    stolen" scores zero regex hits for every type (motor's list needs the
    full phrase "car insurance", not bare "car"; "stolen" isn't "theft")
    and falls through to "general," meaning the query gets no type signal
    at all despite obviously being about motor insurance to a human reader.

    Kept deliberately fast and cheap (few output tokens, short timeout)
    since — unlike the document/chunk-classification LLM calls elsewhere
    in this codebase, which run in a background thread after the ingest
    response has already returned — this sits on the LIVE query's
    critical path: the result feeds directly into the retrieval filter
    below, so it has to resolve before the vector search can even run.
    Falls back to "general" (no filtering) on any failure, exactly like
    the regex path already does.
    """
    try:
        label_list = "\n".join(f"  - {pt}: {info['desc']}" for pt, info in get_active_vocab().items())
        prompt = f"""Classify the ONE insurance policy type this short user question is about.

{label_list}
  - general: the question doesn't name or clearly imply one specific type, or spans several

Don't default to "commercial" just because a question mentions a business,
fleet, company, or workplace context — check whether a MORE SPECIFIC type
fits first (a business's fleet of vans is still motor insurance, a business
being sued by a client is still liability insurance, a business's stock
lost to fire is still fire insurance). Only use "commercial" when the
question is genuinely about the business's own premises or operational
continuity, with no more specific type actually fitting.

Use the EXACT label word above (e.g. "motor", not "car" or "auto"; "home", not "property").

QUESTION: {query}

Reply with ONLY the label word, nothing else."""
        raw = await _backend_completion(prompt, max_tokens=10, timeout=3)
        if not raw:
            return "general"
        label = re.split(r"[\s\n,.:;()]", raw.strip().lower())[0]
        label = _normalize_policy_type(label)
        result = label if label in _valid_policy_types() else "general"
        logger.debug("[QUERY_POLICY_TYPE] LLM fallback: %r -> %s", query, result)
        return result
    except Exception as exc:
        logger.debug("[QUERY_POLICY_TYPE] LLM fallback failed (%s) — treating as general", exc)
        return "general"


async def _classify_query_candidate_type_llm(query: str) -> Optional[str]:
    """
    Open-vocabulary sibling of _classify_query_policy_type_llm() — no list
    constraint, can name anything, even a label the candidate vocabulary has
    never seen before. Never touches retrieval filtering; its only purpose
    is comparison against a chunk's candidate_policy_type during reranking
    (see _effective_sort_score in ask_stream). Callers only invoke this
    when the retrieved pool actually contains a chunk with
    candidate_policy_type set — the common case has nothing to compare
    against, so this call would otherwise be pure wasted latency on every
    single query.
    """
    from candidate_vocab import match_candidate_vocab, normalize_candidate_label, upsert_candidate

    hit = match_candidate_vocab(query)
    if hit:
        upsert_candidate(hit, [], query, "query")
        return hit

    try:
        prompt = f"""This is a short insurance-related question. If it names or clearly
implies ONE specific insurance product or coverage type — even an unusual
or unfamiliar-sounding one — extract that name in 1-3 words. If the
question itself already uses a specific term for the coverage (even a term
you don't recognize as a standard insurance category), use that exact term
rather than judging whether it "sounds like" a real product.

- If yes, reply with ONLY that name, lowercase, 1-3 words, nothing else.
- If the question is generic, spans multiple types, or doesn't name or
  imply one specific product at all, reply with exactly: general

QUESTION: {query}

ANSWER:"""
        raw = await _backend_completion(prompt, max_tokens=10, timeout=3)
        if not raw:
            return None
    except Exception as exc:
        logger.debug("[CANDIDATE_TYPE] query open-ended LLM call failed: %s", exc)
        return None

    label = normalize_candidate_label(raw)
    if label is None:
        return None
    upsert_candidate(label, [], query, "query")
    logger.debug("[CANDIDATE_TYPE] query open-ended guess: %r -> %r", query, label)
    return label


# Context passed to the grounding-check prompt is capped so this stays a
# fast, cheap call (max_tokens=10 output either way) — not specified by the
# original design, but consistent with the context[:1000] pattern elsewhere
# in this file bounding auxiliary-call input size for latency.
#
# Doubled 3000 -> 6000 alongside _build_grounding_context's window-joining
# fix (see that function's docstring/comment) — that fix doubled how much
# text each chunk contributes (both _rerank_windows candidates instead of
# arbitrarily just one), which at the OLD 3000 cap roughly halved how many
# of the already-best-first-sorted chunks actually fit before truncation,
# silently losing lower-ranked-but-still-relevant chunks that used to be
# visible. Confirmed live: "What is the maximum compensation for legal
# expenses under travel insurance?" needs a phrase from the chunk ranked
# #3 (the word "legal", establishing which of two adjacent, identically-
# shaped clauses a figure in chunk #1 belongs to) — at the old cap that
# phrase sat at char ~3849, past the truncation point, even though it was
# genuinely present in the retrieved pool. Doubling the cap alongside the
# per-chunk content restores roughly the same number of visible chunks as
# before the window-joining fix, so this isn't a net increase in how much
# "extra" content the model sees per chunk actually retrieved — only a
# correction to keep the same chunk-count budget intact.
_GROUNDING_CONTEXT_CHARS = 6000


def _build_grounding_context(query: str, chunks: list) -> str:
    """Join *chunks* into a single preview string for _verify_grounding(),
    using each chunk's most query-relevant excerpt (via _rerank_windows,
    already computed cheaply during reranking with no extra model call)
    instead of its raw, full page_content.

    Confirmed live: with several full multi-hundred-word chunks joined and
    then hard-capped at _GROUNDING_CONTEXT_CHARS, a genuinely answering
    sentence sitting past the opening of a long chunk gets buried among
    unrelated boilerplate (exam questions, generic definitions, other
    chunks) ahead of it in the same budget — "How are relatives typically
    notified about the contents of a policy document?" against a context
    that DID contain "...kept in safe custody and in the knowledge of the
    close relatives" still got NO from the grounding check, but the exact
    same sentence, isolated from the surrounding noise, reliably got YES.
    Using each chunk's best-matching window keeps the signal-to-noise ratio
    high regardless of where in a long chunk the relevant sentence sits —
    the same fix already proven for the reranker's own scoring step.
    """
    parts = []
    for c in chunks:
        text = getattr(c, "page_content", "") or ""
        if not text:
            continue
        windows = _rerank_windows(text, query)
        # Used to take ONLY windows[1] (the keyword-weighted window),
        # discarding windows[0] (the plain start-of-chunk slice) whenever
        # both existed. That's an arbitrary fixed-index choice, not "pick
        # whichever window actually scored higher" — _rerank_windows'
        # caller (the real reranking pass) takes the MAX score across
        # windows, but that per-window score isn't threaded through to
        # here, so this function had no way to know which one actually
        # mattered. Confirmed live: "What is the maximum compensation for
        # legal expenses under travel insurance?" against a chunk whose
        # relevant sentence ("...the maximum amount of compensation is
        # €8,500...") sits right at the start (windows[0]) — the
        # keyword-weighted windows[1] drifted past it to an adjacent,
        # differently-scoped clause ("...liability...€170,000...") that
        # scores similarly on keyword density but doesn't answer the
        # question at all. Dropping windows[0] unconditionally threw away
        # the one window that actually had the answer, and the reranker's
        # own 0.976 score (which DOES account for windows[0]) never gets a
        # chance to inform this check. Joining both windows instead of
        # picking one fixes this without needing to plumb the winning
        # window's score through from reranking — bounded to ~1400 chars
        # per chunk (2 windows of length _RERANK_TEXT_CHARS=700), and since
        # chunks arrive sorted best-first (this function's caller already
        # requires that), the top chunk's full ~1400 chars land within the
        # first slice of _GROUNDING_CONTEXT_CHARS=3000 regardless of what
        # lower-ranked chunks contribute afterward. This is a different
        # tradeoff than the ORIGINAL single-full-chunk-per-entry approach
        # this function replaced (see docstring above) — that discarded
        # windowing entirely and diluted signal with multiple full,
        # uncapped chunks; two short curated windows per chunk doesn't
        # reintroduce that problem.
        parts.append("\n".join(windows))
    return "\n\n".join(parts)


async def _verify_grounding(question: str, context: str, backend_override: Optional[str] = None) -> bool:
    """LLM-based semantic grounding backstop — an authoritative layer on top
    of the lexical/regex coverage checks (_context_covers_query,
    _enumeration_query_covered, _quoted_comparison_covered), not a
    replacement for them.

    Those lexical checks are pattern-matching heuristics: every new
    phrasing that slips past them needs a new regex rule. This closes that
    gap by directly asking the model whether the SPECIFIC question is
    answerable from the SPECIFIC retrieved context — which catches the
    failure mode the lexical checks structurally can't: a chunk that's
    topically adjacent but not actually about the thing asked (e.g. a
    Takaful-insurance-model chunk confidently used to answer "which
    insurers cover travel to South Africa" — same broad category
    (insurance), wrong specific topic, no lexical rule can enumerate every
    such near-miss in advance).

    backend_override: see _backend_completion — used by the query-cleaning
    fallback path to re-check grounding on vLLM specifically, so that path
    stays independent of Groq's daily quota (the whole reason it exists).

    Fail-safe: any exception, timeout, or ambiguous/empty response returns
    False (not grounded) — matches the fail-safe direction already used
    elsewhere in this file for needs_human (when in doubt, refuse rather
    than risk answering ungrounded).
    """
    if not context or not context.strip():
        return False
    # Framed as "is there relevant info here" rather than "can this EXACT
    # question be answered" — the earlier "exact question" framing made the
    # 7B model demand something close to a literal restatement of the
    # question in the context. That broke down specifically on totality/
    # completeness questions ("are all types of illness covered?") against
    # a long, multi-topic joined context: an exclusions list is exactly the
    # right way to answer such a question (implies "no, not everything —
    # here are the exceptions"), but the model wouldn't credit it without
    # an explicit "not all are covered" sentence to point to. Confirmed via
    # a 3x-repeated, 4-case comparison (this exact case + 3 known-good
    # regression cases) that the "exact question" framing failed the
    # exclusions case 100% of the time regardless of added exclusion-list
    # guidance, while this framing passes it and still correctly returns NO
    # on genuinely wrong-topic context (Takaful/South-Africa mismatch case).
    prompt = (
        f"Context:\n{context[:_GROUNDING_CONTEXT_CHARS]}\n\n"
        f"Question: {question}\n\n"
        "Does this context contain information directly relevant to "
        "answering this question — enough that someone could give a real, "
        "specific answer (not necessarily complete or exhaustive)? Answer "
        "NO if: the context is about a different, unrelated topic; the "
        "question asks for a specific fact (a country, provider, number) "
        "that is simply absent; OR the question describes a specific "
        "triggering event or scenario (e.g. a policy reaching its natural "
        "end date without any claim made) and the context only discusses a "
        "DIFFERENT triggering event in the same general area (e.g. "
        "cancelling a policy early, or returning it during an initial "
        "review window) — sharing a general subject like \"refunds\" is not "
        "the same as the context actually addressing the specific event "
        "asked about. This does not apply to a list that implies its own "
        "completeness (e.g. an exclusions list correctly implies everything "
        "not listed IS covered) — that is still a real answer, not a "
        "different scenario. "
        "Answer with a single word: YES or NO."
    )
    try:
        raw = await _backend_completion(prompt, max_tokens=10, timeout=4.0, backend_override=backend_override)
        if not raw:
            return False
        cleaned = re.sub(r"[^a-z\s]", "", raw.strip().lower())
        words = set(cleaned.split())
        return "yes" in words and "no" not in words
    except Exception:
        return False


async def _verify_grounding_any_chunk(
    question: str, chunks: list, backend_override: Optional[str] = None,
) -> bool:
    """Grounded if EITHER the full joined multi-chunk context grounds the
    question OR the single top-ranked chunk alone does — checked in
    parallel, not sequentially, so this costs no extra latency over the
    plain multi-chunk check on the common (already-passing) path.

    Chunks must already be sorted best-first (true of every caller — all
    come from _rerank()/rerank_documents() output).

    Confirmed live: joining even one additional, lower-relevance chunk
    alongside the correct one can flip _verify_grounding()'s YES to a NO
    for the WHOLE block — "How are relatives typically notified about the
    contents of a policy documents?" against just the top chunk (containing
    "...kept in safe custody and in the knowledge of the close relatives")
    returned YES in isolation, but adding a second, merely-topically-
    adjacent chunk (about policy document contents generally) flipped the
    combined judgment to NO, even though the first chunk's relevant content
    didn't change. This is a real limitation of this codebase's small model
    when judging heterogeneous multi-chunk blocks (see _reformulate_query's
    docstring for other confirmed instances of this model's unreliability
    on nuanced multi-part judgments) — checking the strongest single
    candidate on its own recovers the case without weakening the existing
    multi-chunk check, which still runs and still catches genuinely
    wrong-topic content (verified: an off-topic single chunk alone is still
    correctly rejected, so this isn't a blanket loosening).
    """
    if not chunks:
        return False
    full_context = _build_grounding_context(question, chunks)
    top_context = _build_grounding_context(question, chunks[:1])
    full_result, top_result = await asyncio.gather(
        _verify_grounding(question, full_context, backend_override=backend_override),
        _verify_grounding(question, top_context, backend_override=backend_override),
    )
    return full_result or top_result


async def _contextualize_query(question: str, history: str) -> str:
    """Rewrite the question into a standalone, self-contained form by
    resolving pronouns and implicit references against recent conversation
    history. Runs on every turn — a true first-turn or already-standalone
    question should be returned unchanged, not gated behind a separate
    followup/not-followup classifier.

    *history* is a flat "User: ...\\nAssistant: ..." string as built by
    ConversationAgent._build_history_string(). Uses only the last 1-2
    turns via _split_history_turns().

    Fast-path: if the question contains no reference token at all, it's
    structurally standalone — skip the LLM call entirely. This is a
    latency optimization only; the LLM prompt below is what actually
    enforces correctness (a false-positive regex match just costs one
    extra LLM call that correctly returns the question unchanged).

    Fail-safe: on any exception, timeout, or empty response, return the
    original question unchanged.
    """
    if not history or not history.strip():
        return question

    if not _REFERENCE_TOKENS.search(question.strip().lower()):
        return question

    lines = _split_history_turns(history)
    recent = lines[-4:]
    if not recent:
        return question
    history_text = "\n".join(recent)

    prompt = (
        f"Recent conversation:\n{history_text}\n\n"
        f"New question: {question}\n\n"
        "Does the new question contain a pronoun or implicit reference "
        "(e.g. 'it', 'that', 'those', 'their', 'the second one') that "
        "depends on the conversation above to be understood?\n"
        "If YES, rewrite the question to resolve that reference, "
        "replacing the pronoun/reference with the specific thing it "
        "refers to. If the reference is to an ordinal position in a "
        "numbered or listed answer above (e.g. 'the second point', "
        "'point 3', 'the last one'), rewrite it to name the SPECIFIC "
        "subject of that one point only — do not fold in neighboring "
        "points.\n"
        "If NO — the question is already a complete, standalone "
        "question, even if it's on a different topic than the "
        "conversation above — return the question completely "
        "UNCHANGED. Do not add topic context to a question that "
        "doesn't need it.\n"
        "Respond with ONLY the question (rewritten or unchanged), "
        "nothing else."
    )
    try:
        raw = await _backend_completion(prompt, max_tokens=60, timeout=4.0)
        if not raw or not raw.strip():
            return question
        return raw.strip()
    except Exception:
        return question


async def _extract_pasted_followup(question: str) -> Tuple[Optional[str], str]:
    """Detect a message that's really "<a block of previously-given text>"
    plus "<a short question about it>" pasted together as one message — e.g.
    a user copying a chunk of an earlier answer (even one from many turns
    back, well outside the retained history window) and asking about it
    directly, rather than relying on the system to still remember that far.

    Returns (pasted_context, actual_question). pasted_context is None (and
    actual_question is the original, unmodified question) when no paste is
    detected, including on any failure or ambiguity — the caller then
    behaves exactly as it did before this function existed.
    """
    # An ordinary question is never this long — skip the LLM round-trip
    # entirely below the bar where a paste becomes plausible.
    if len(question.split()) < 40:
        return None, question
    prompt = (
        f"Message:\n{question[:6000]}\n\n"
        "This message may be a block of previously-given text (e.g. copied "
        "from an earlier answer) followed by a short question or "
        "instruction about it (e.g. 'explain this simply', 'summarize the "
        "above', 'what does point 3 mean'). If so, respond with ONLY that "
        "short question or instruction, copied VERBATIM from the message — "
        "nothing else. If the message is just one ordinary question with "
        "no pasted block, respond with exactly: NONE"
    )
    try:
        raw = await _backend_completion(prompt, max_tokens=60, timeout=4.0)
    except Exception:
        return None, question
    if not raw or not raw.strip():
        return None, question
    extracted = raw.strip()
    if extracted.upper() == "NONE":
        return None, question
    idx = question.rfind(extracted)
    if idx == -1:
        # Couldn't confidently locate the extracted question verbatim in the
        # original message — be conservative and treat this as no paste
        # rather than guessing at a split point.
        return None, question
    pasted = (question[:idx] + question[idx + len(extracted):]).strip()
    # The remaining "pasted" portion must itself be substantial, or this was
    # likely just an ordinary question the model over-matched on.
    if len(pasted.split()) < 20:
        return None, question
    return pasted, _strip_paste_reference_filler(extracted)


# A follow-up about pasted content naturally opens with a connector back to
# it ("now based on this,", "given the above,") — these add no topical
# signal of their own and measurably dilute retrieval quality (embedding
# score for "what is the free look period..." dropped from 0.34 to 0.06
# once such a prefix was added, in testing). Stripped only from the
# extracted question, not from pasted_context or the user-facing answer —
# the LLM answering the question still has pasted_context available to
# resolve "this" against; only the KB search string needs to be clean.
_PASTE_FILLER_RE = re.compile(
    r'^\s*(?:now|so|then|well|okay|ok)?[,\s]*'
    r'(?:based on (?:this|that|the above)|given (?:this|that|the above)(?: information)?|'
    r'considering (?:this|that|the above))'
    r'(?: information)?[,\s]*',
    re.IGNORECASE,
)


def _strip_paste_reference_filler(text: str) -> str:
    stripped = _PASTE_FILLER_RE.sub('', text).strip()
    return stripped if stripped else text


# ── Ordinal point-reference follow-ups ("explain point 2", "the 2nd point") ──
# _contextualize_query() resolves these into a standalone retrieval query, but
# that resolution is itself unreliable: sometimes the model substitutes the
# point's real subject matter (works), sometimes it only normalizes the
# wording ("point 2" -> "the second point", still no topical content), and
# retrieval then has nothing to search with — confirmed in testing on two
# back-to-back examples that only differed in which the model happened to
# produce. The most reliable source for "what was point 2 about" is the
# numbered list already sitting in conversation history, not a fresh KB
# search trying to rediscover the same content. _extract_point_reference()
# pulls that point's own text out of history and feeds it through the exact
# same pasted-context path as a live paste (verify-then-bypass-or-blend) —
# so this reuses tested machinery rather than adding a parallel one.
_ORDINAL_WORDS = {
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'fifth': 5,
    'sixth': 6, 'seventh': 7, 'eighth': 8, 'ninth': 9, 'tenth': 10,
}
_POINT_REFERENCE_RE = re.compile(
    r'\bpoint\s+number\s+(\d+)\b|'
    r'\bpoint\s+(\d+)\b|'
    r'\bthe\s+(\d+)(?:st|nd|rd|th)?\s+point\b|'
    r'\bthe\s+(' + '|'.join(_ORDINAL_WORDS) + r')\s+point\b|'
    # Bare "N point" with no leading "the" — e.g. "explain 2 point in detail".
    # Confirmed live: "can you explain 2 point in simple language with
    # example" fell through every existing alternative above (all require
    # either "point N" ordering or a leading "the"), so _extract_point_number
    # returned None, point-text extraction never ran, and the question
    # reached the standalone-retry tier as an apparently topic-less query.
    r'\b(\d+)(?:st|nd|rd|th)?\s+points?\b|'
    r'\bnumber\s+(\d+)\b|'
    r'\bthe\s+(\d+)(?:st|nd|rd|th)\b(?!\s+\w)',
    re.IGNORECASE,
)
_NUMBERED_POINT_RE = re.compile(r'(?:^|\n)\s*(\d+)\.\s+')


def _extract_point_number(question: str) -> Optional[int]:
    """Return the 1-based point number *question* refers to (digit or
    spelled-out ordinal), or None if it doesn't reference one at all."""
    m = _POINT_REFERENCE_RE.search(question.lower())
    if not m:
        return None
    for g in m.groups():
        if g is None:
            continue
        if g.isdigit():
            return int(g)
        if g in _ORDINAL_WORDS:
            return _ORDINAL_WORDS[g]
    return None


def _extract_point_text_from_history(history: str, point_num: int) -> Optional[str]:
    """Find point *point_num*'s own text in the MOST RECENT assistant turn's
    numbered list — None if that turn isn't numbered, or that point number
    isn't in it.

    Deliberately does NOT keep searching older turns when the most recent
    answer has no numbered list. Used to `continue` past it and grab a
    numbered list from further back in history — confirmed live: the most
    recent answer used a "Term: description" glossary-style format with no
    digit markers (a real, separate formatting drift, tracked elsewhere),
    so this fell through to a numbered list from an unrelated, much earlier
    turn in the same long-running session (a property/fire insurance
    example) and fed it into a follow-up about the CURRENT health insurance
    answer — wrong-topic content injected as if it were "point 2" of what
    the user was actually looking at. "Point 2" always means point 2 of the
    answer on screen right now; if that answer isn't numbered, there's no
    point 2 to find, full stop — don't guess from history.
    """
    turns = _split_history_turns(history)
    for turn in reversed(turns):
        if not turn.startswith("Assistant:"):
            continue
        content = turn[len("Assistant:"):].strip()
        matches = list(_NUMBERED_POINT_RE.finditer(content))
        if not matches:
            return None
        for i, m in enumerate(matches):
            if int(m.group(1)) == point_num:
                start = m.end()
                end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
                point_text = content[start:end].strip()
                return point_text or None
        return None
    return None


_MEANING_QUERY_RE = re.compile(
    r"^(?:"
    r"what\s+(?:does|do|is)\s+(?:the\s+word\s+|the\s+term\s+)?"
    r"['\"]?(?P<w1>[a-z][a-z\s'-]{0,40}?)['\"]?\s+mean\b"
    r"|what\s+do\s+you\s+mean\s+by\s+['\"]?(?P<w2>[a-z][a-z\s'-]{0,40}?)['\"]?\s*\??$"
    r"|what\s+is\s+the\s+meaning\s+of\s+['\"]?(?P<w3>[a-z][a-z\s'-]{0,40}?)['\"]?\s*\??$"
    r"|meaning\s+of\s+['\"]?(?P<w4>[a-z][a-z\s'-]{0,40}?)['\"]?\s*\??$"
    r"|define\s+['\"]?(?P<w5>[a-z][a-z\s'-]{0,40}?)['\"]?\s*\??$"
    r"|what\s+does\s+['\"]?(?P<w6>[a-z][a-z\s'-]{0,40}?)['\"]?\s+stand\s+for\b"
    r")",
    re.IGNORECASE,
)

# Pronouns/generic fillers the regex above will happily capture as "the
# word" (e.g. "what does that mean?" → w1="that") but that are never what
# the user actually wants defined — that phrasing means "explain the thing
# you just said differently", handled elsewhere by the meta-clarify and
# contextualization paths, not "look up the dictionary meaning of 'that'".
_MEANING_QUERY_STOPWORDS = frozenset({
    "that", "this", "it", "they", "them", "he", "she", "we", "you", "i",
    "something", "anything", "everything", "nothing", "one", "these", "those",
})


def _extract_meaning_query_word(question: str) -> Optional[str]:
    """Return the word/short phrase *question* is asking the meaning of
    ("what does X mean?", "meaning of X", "define X", "what does X stand
    for?"), or None if it isn't that kind of question.

    Deliberately narrow (anchored patterns, not a loose "contains 'mean'"
    check) so it doesn't fire on unrelated sentences that happen to contain
    the word "mean" (e.g. "what does this mean for my premium" — a real
    coverage question, not a word-definition request). Also filters out
    pronouns/fillers via _MEANING_QUERY_STOPWORDS, since the regex alone
    can't distinguish "what does 'discount' mean?" (real word) from "what
    does that mean?" (pronoun — regex would otherwise capture "that").
    Callers should still verify the extracted word actually appears in
    recent history before using it, same discipline as
    _extract_point_number's callers.
    """
    m = _MEANING_QUERY_RE.match(question.strip())
    if not m:
        return None
    for name in ("w1", "w2", "w3", "w4", "w5", "w6"):
        w = m.group(name)
        if w and w.strip() and w.strip().lower() not in _MEANING_QUERY_STOPWORDS:
            return w.strip()
    return None


_HANDOFF_MSG = (
    "I don't have that in my knowledge base right now. "
    "Let me get a human agent to help you! 😊"
)


def _strip_markdown(text: str) -> str:
    """Convert markdown-formatted LLM output to plain conversational prose.

    The chat prompt forbids bullet points and bold, but the model sometimes
    ignores that — especially when the retrieved context itself contains
    formatted content. Called by api.py's streaming handler on each small
    flushed chunk of the response as it streams (not the whole response at
    once), so the user never sees raw markdown mid-stream.

    Deliberately does NOT strip numbered-list markers ("1. ", "2. ") the
    way it used to — DETAILED_GROUNDED_PROMPT explicitly wants numbered
    points, and ask_stream() already handles both directions correctly on
    the complete response before this ever runs: it enforces numbering
    deterministically when a detailed answer's model output didn't use it,
    and strips stray markers when a brief answer's shouldn't have any (see
    multi_source_rag.py's "Strip stray numbered-list markers" section).
    Stripping markers here too, on arbitrary small per-flush chunks, was
    actively harmful rather than redundant: since api.py flushes as soon as
    a chunk contains ANY space or newline, whether "\n2. " for a genuinely
    correct detailed-mode point landed inside the SAME flush as its
    trailing space (letting this regex match and destroy it) or got split
    across two separate flushes (leaving it untouched) came down to
    arbitrary token-boundary luck for that specific generation — confirmed
    live: the exact same "explain in detail" request rendered as a clean
    numbered list in some runs and a single unbroken paragraph in others,
    with no difference in the underlying model output's actual formatting.
    That, in turn, silently broke follow-up "explain point N" references,
    since the numbering that later parsing depends on had already been
    destroyed before it was ever saved to conversation history.
    """
    import re
    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # Remove ATX headers (## Heading -> Heading)
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    # Convert bullet list items to flowing prose: "- item" or "* item" -> "item, "
    text = re.sub(r'\n\s*[-*]\s+', ' ', text)
    # Inline code backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Collapse 3+ newlines -> 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def _strip_model_preamble(text: str) -> str:
    """Remove auto-generated meta-commentary lines the LLM prepends to answers."""
    # Use simple string replace to strip robot emoji — avoids regex encoding issues
    text = text.replace('\U0001F916', '').replace('\U0001f916', '')
    _TEXT_STARTS = (
        "response was brief",
        "no specific values or formulas",
        "no further action was needed",
    )
    lines = text.split("\n")
    clean = [l for l in lines if not any(l.strip().lower().startswith(p) for p in _TEXT_STARTS)]
    # Remove leading blank lines
    while clean and not clean[0].strip():
        clean.pop(0)
    return "\n".join(clean).strip()


_RULE4_MARKER_RE = re.compile(r"i don.t have that specific", re.IGNORECASE)

# Matches the filler-word usage of "honest,"/"honestly," (comma right after,
# functioning as a sentence-starting interjection) — not a legitimate
# adjective use like "an honest answer". The prompt now bans this word
# outright, but see the ask_stream call site for why that alone isn't
# trusted: this filler was already a known compliance gap when the prompt
# only asked for the full "honestly," spelling (the model unreliably
# shortened it to "honest,"), so banning the word entirely needs the same
# deterministic backstop, not just the new instruction. Only matches at a
# sentence boundary (start of string, after ./!/?  + space, or after a
# newline) so it strips the discourse-marker usage and capitalizes what
# follows, without touching "an honest answer" mid-sentence.
_HONESTLY_FILLER_RE = re.compile(r"(?:^|(?<=[.!?]\s)|(?<=\n))[Hh]onest(?:ly)?,\s*(\w)")

# The frontend renders assistant messages as plain text (no markdown
# parser) — confirmed live: the model sometimes writes "**Legal
# Expenses**: ..." for emphasis, which the chat UI shows as literal
# asterisks around the words instead of bold text. Stripped
# deterministically rather than adding markdown rendering to the
# frontend, consistent with how this session has handled every other
# case of "the model does something the display layer doesn't expect."
# Non-greedy and requires the pair to close on the same line so it
# can't accidentally swallow unrelated text if the model ever emits an
# unpaired "**".
_MARKDOWN_BOLD_RE = re.compile(r"\*\*([^\n*]+?)\*\*")

# Every live prompt explicitly bans the em dash (—), but the model doesn't
# reliably follow that — same lesson as every other formatting-compliance
# gap in this file (numbered lists, the "honestly" filler, markdown bold):
# don't trust a prompt-only instruction, enforce it deterministically too.
# Confirmed live: "wat is a deductable in helth insurnace" (a fresh, never-
# tested typo-laden query) came back with "It's like setting a threshold —
# you handle the cost..." despite the ban. Replaced with a comma rather
# than a period — safer as a universal default since guessing "does this
# em dash join two independent clauses" wrong would produce a sentence
# split with broken capitalization; a comma splice reads fine in the
# casual, conversational tone this bot already uses everywhere.
_EM_DASH_RE = re.compile(r"\s*—\s*")

# The vLLM backend (Qwen2.5-7B) occasionally code-switches into Chinese
# mid-response even though the question and the rest of the answer are
# English — confirmed live: "...plus any bonuses累积的。具体金额取决于
# 你的保单条款..." tacked onto an otherwise-complete English sentence.
# Re-running the identical question 4/4 times came back clean English,
# so this is a non-deterministic generation-sampling artifact, not a
# retrieval or prompt bug worth chasing upstream. Stripped
# deterministically like every other display-layer mismatch this
# session: delete the non-Latin-script run and clean up what's left
# rather than showing garbled mixed-language text. Covers CJK ideographs
# plus Japanese kana, Hangul, and CJK/fullwidth punctuation so it isn't
# tied to this one Chinese-specific incident.
_NON_LATIN_SCRIPT_RE = re.compile(
    r"[　-〿぀-ヿ㐀-䶿一-鿿가-힯＀-￯]+"
)


def _strip_rule4_fallback(text: str, trust_content: bool = True) -> Optional[str]:
    """Handle the LLM appending the canned Rule 4 fallback ("Hmm, I don't
    have that specific info...") after already writing something, or
    putting the refusal marker FIRST and real content AFTER it.

    Returns:
      None            — marker not found, caller should leave text untouched.
      non-empty str   — marker found, trust_content was True, and there was
                         real content on at least one side of the marker
                         (>40 chars) — return the best trustworthy content
                         (before, after, or both joined).
      ""              — marker found but trust_content was False (or neither
                         side has enough content) — the marker is the model's
                         own admission the text isn't solidly grounded.
                         Caller should discard and show the standard refusal.

    trust_content should reflect independent evidence (e.g. a high reranker
    score) that the content is actually grounded — text pattern matching
    alone can't tell a correct answer with a pointless disclaimer apart from
    a shaky inference with a legitimate one.

    Applied both to freshly-generated answers AND to KV cache hits — a cache
    hit can replay an answer that was cached before this fix existed, so the
    check has to run on served text either way, not just on fresh generations.
    """
    stripped = (text or "").rstrip()
    m = _RULE4_MARKER_RE.search(stripped)
    if not m:
        return None

    real_before = stripped[:m.start()].strip()

    # The refusal sentence itself can run several sentences past the
    # marker start (e.g. "...I don't have that specific info right now.
    # Let me get one of our agents on it!") before real content resumes.
    # Find where the NEXT sentence after the marker begins so we don't
    # keep fragments of the refusal itself in real_after.
    tail = stripped[m.end():]
    # Skip to the first sentence boundary in the tail (end of the
    # refusal sentence/clause), then trim any immediately-following
    # generic handoff filler sentence if present.
    sentence_end = re.search(r'[.!?]\s+', tail)
    real_after = tail[sentence_end.end():].strip() if sentence_end else ""
    # Drop a leading handoff-filler sentence from real_after if the model
    # chained it right after ("...Let me get one of our agents on it...")
    handoff_lead = re.match(
        r'(let me get (?:one of our agents|a human agent)[^.!?]*[.!?]\s*)',
        real_after, re.IGNORECASE,
    )
    if handoff_lead:
        real_after = real_after[handoff_lead.end():].strip()

    if not trust_content:
        return ""

    if len(real_before) > 40 and len(real_after) > 40:
        return f"{real_before} {real_after}".strip()
    if len(real_before) > 40:
        return real_before
    if len(real_after) > 40:
        return real_after
    return ""


# ── Short follow-up detection & reformulation ──────────────────────────────
# When a user sends a very short message (e.g. "yes", "ok", "what about that")
# the raw text has no standalone semantic content for vector retrieval, causing
# near-zero similarity scores and a false human-handoff trigger.  If history is
# available we reformulate the query by merging with the last assistant turn.
_SHORT_FOLLOWUP_PHRASES = frozenset({
    "yes", "ok", "okay", "sure", "yeah", "yep", "yup",
    "tell me more", "go on", "continue", "and", "and?",
    "what about that", "what about it", "how about that", "how about it",
    "explain", "elaborate", "more", "really", "interesting",
    "i see", "got it", "understood", "right", "correct",
    "no", "nope", "nah", "not really",
    "why", "why not", "how", "how so",
    "can you elaborate", "can you explain",
})

# Referential/ordinal patterns that indicate a follow-up question even when
# the message is longer than 4 words (e.g. "explain about the second one more").
# These carry no standalone retrieval meaning — they refer back to items the
# assistant just listed — so they must trigger history-based reformulation.
_REFERENTIAL_PATTERN = re.compile(
    r"(?:"
    r"\b(first|second|third|fourth|fifth|last|other)\s+(one|option|type|item)\b|"
    r"\b(that|this)\s+one\b|"
    r"\bthe\s+other\s+one\b|"
    r"\bmore\s+about\s+(it|that|this)\b|"
    r"\bexplain\s+(more\s+)?(about\s+)?the\s+(first|second|third|fourth|fifth|last|other)\b|"
    r"\btell\s+me\s+more\s+about\s+the\s+(first|second|third|fourth|fifth|last|other)\b|"
    r"\belaborate\s+on\s+that\b|"
    r"\bexplain\s+more\b"
    r")"
)


def _is_short_followup(question: str) -> bool:
    """Detect if *question* is a short, low-content follow-up message.

    Returns ``True`` when the message is under ~4 words and either matches a
    known continuation phrase or is so short it cannot carry standalone
    retrieval meaning.  Also returns ``True`` for longer messages that match
    referential/ordinal patterns (e.g. "explain about the second one more")
    because those refer back to items the assistant just listed and have no
    standalone retrieval value.
    """
    q = question.strip().lower().strip("!.,?;:")
    words = q.split()

    # Referential/ordinal patterns — independent of word count.
    # These questions refer back to items the assistant just listed and have
    # no standalone retrieval meaning (e.g. "explain about the second one").
    if _REFERENTIAL_PATTERN.search(q):
        return True

    if len(words) > 4:
        return False
    # Exact match against known continuation phrases
    if q in _SHORT_FOLLOWUP_PHRASES:
        return True
    # Any 1-2 word message that isn't a normal-length question qualifies
    if len(words) <= 2:
        return True
    return False


# A flat history string is "User: ...\nAssistant: ...\nUser: ...\nAssistant: ..."
# — but a single turn's own content can contain internal newlines (e.g. a
# numbered-list answer, one "\n" per point). A naive history.split("\n") to
# grab "the last N lines" fragments that ONE long turn into many pieces, so
# the window can end up holding only the tail of a numbered list (points
# 6-8) while dropping the earlier points (1-5) entirely — and a follow-up
# asking about "point 2" then has no way to know what point 2 actually was,
# since it was never in the history it saw at all (this exact regression
# previously shipped and was fixed once already — see git history on this
# function — before being silently reintroduced by an unrelated merge).
# Splitting only at newlines immediately followed by "User:"/"Assistant:"
# keeps each turn whole no matter how many lines its own content spans.
_HISTORY_TURN_BOUNDARY_RE = re.compile(r'\n(?=(?:User|Assistant):\s)')


def _split_history_turns(history: str) -> list[str]:
    """Split a flat ``"User: ...\\nAssistant: ..."`` history string into whole
    turns (see module note above), not naive newline-delimited lines — so
    callers can slice the last N turns without cutting a multi-line response
    in half.
    """
    return [t.strip() for t in _HISTORY_TURN_BOUNDARY_RE.split(history.strip()) if t.strip()]


def _last_user_question(history: str) -> str:
    """Text of the most recent USER turn in *history*, "User: " prefix stripped."""
    for turn in reversed(_split_history_turns(history)):
        if turn.startswith("User:"):
            return turn[len("User:"):].strip()
    return ""


def _history_last_assistant_turn(history: str) -> str:
    """Text of the most recent ASSISTANT turn in *history*, prefix stripped.

    Named distinctly from the unrelated local variable `_last_assistant_turn`
    already used inside ask_stream() — Python's local-scope rules mean a
    same-named local variable anywhere in that function shadows a module-
    level function of the same name for the function's entire body, which
    broke this call with `TypeError: 'str' object is not callable` when
    first written as `_last_assistant_turn`.
    """
    for turn in reversed(_split_history_turns(history)):
        if turn.startswith("Assistant:"):
            return turn[len("Assistant:"):].strip()
    return ""


# A generic "elaborate on whatever we were just discussing" follow-up, as
# opposed to a follow-up that names its own specific new content ("how do I
# claim it?", "is it tax deductible?"). Only THIS class of follow-up is
# ambiguous about *which part* of a compound prior question the user wants
# more on — a substantive follow-up already says what it's asking about.
_ELABORATION_FOLLOWUP_RE = re.compile(
    r"\b(explain.{0,15}in detail|more detail|tell me more|elaborate|"
    r"go (?:into|in)\s+detail|detailed explanation|explain (?:it|that|this)\b)",
    re.IGNORECASE,
)

# Matches messages that START WITH one of these phrases — not a full-string
# match — because the "Both" clarify chip's own label ("Both — give me the
# full picture", set below in _CLARIFY_OPTIONS suffix) is sent verbatim as
# the user's message when clicked. An exact full-phrase match (the original
# design) missed that: "Both — give me the full picture" starts with "both"
# but has trailing text a `$`-anchored pattern rejects. Still narrow enough
# not to false-positive on a genuine question — real questions don't
# normally open with "both"/"everything" as their first word.
_BOTH_REPLY_RE = re.compile(
    r"^\s*(both\b|all of them\b|everything\b|"
    r"(?:give|answer|tell)\s+me\s+both\b|i\s+want\s+both\b)",
    re.IGNORECASE,
)

# Distinctive enough to reliably identify our own clarification message in
# history (used to recover the two sub-questions for a "both" reply) without
# false-matching on an ordinary answer that happens to contain "both".
_CLARIFY_TEMPLATE = (
    'Happy to go deeper. Did you want more on "{q1}" or "{q2}"? '
    'Click one below, or just say "both" for the full picture on each.'
)
_CLARIFY_PARSE_RE = re.compile(r'more on "(.+?)" or "(.+?)"\?', re.IGNORECASE)

# Structural signal for "this question asks two separate things": a
# wh-word/auxiliary-verb question trigger, then "and", then ANOTHER such
# trigger within a short span — "What is X and what does X cover" has two
# (what...and what), "What is home and fire insurance" has only one (the
# "and" just joins two nouns). Deliberately NOT delegated to an LLM
# yes/no judgment call: tested live, the model classified "What is fire
# insurance and what does it covers?" as NO despite it being nearly
# identical to a worked YES example in the prompt — the same
# unreliable-at-binary-classification pattern seen elsewhere this session
# (e.g. the grounding-check prompt). A structural check is deterministic
# and, unlike the model, actually gets this right.
_COMPOUND_STRUCTURE_RE = re.compile(
    r"\b(what|how|when|where|why|does|do|did|is|are|was|were|can|could|will|would|should)\b"
    r".{2,60}?\band\b.{0,20}?"
    r"\b(what|how|when|where|why|does|do|did|is|are|was|were|can|could|will|would|should)\b",
    re.IGNORECASE,
)


async def _split_compound_question(question: str) -> Optional[tuple[str, str]]:
    """If *question* asks two genuinely separate, independently-answerable
    things (not just two nouns/topics mentioned together — "home and fire
    insurance" is one ask about two topics, not two asks), return the two
    as complete standalone questions. Otherwise return None.

    The IS-it-compound decision is the deterministic regex above, not the
    LLM — only the SPLIT (rephrasing into two clean standalone questions,
    resolving any pronoun) is delegated to the model, which this session
    has repeatedly shown to be reliable at rephrasing but not at binary
    classification.
    """
    if not _COMPOUND_STRUCTURE_RE.search(question):
        return None
    # Ends with a plain instruction, not a "continue after this label" cue —
    # tested live: a trailing "\nLine 1:" completion-style prompt confused
    # this chat model into echoing "Line 1:"/"Line 2:" back garbled into the
    # answer text itself ('What is fire insurance? Line 1: \n\nWhat does it
    # cover? Line 2:'). The plain-instruction style already proven reliable
    # for other multi-line LLM outputs in this file works cleanly here too.
    prompt = (
        f"Question: {question}\n\n"
        "Split this into its two separate questions. Resolve any pronoun so "
        "each question stands alone with no other context needed.\n"
        "Output only the two questions, one per line, no numbering, no labels, no explanation:"
    )
    raw = await _backend_completion(prompt, max_tokens=60, timeout=4.0)
    if not raw:
        return None
    lines = [ln.strip().strip('"') for ln in raw.strip().split("\n") if ln.strip()]
    lines = [re.sub(r"^(line\s*\d\s*:\s*|\d+[.)]\s*)", "", ln, flags=re.IGNORECASE) for ln in lines]
    if len(lines) != 2:
        return None
    if not lines[0].endswith("?") or not lines[1].endswith("?"):
        return None
    return (lines[0], lines[1])


_REFORMULATE_TOPIC_SNIPPET_CHARS = 120


def _reformulate_with_history(question: str, history: str) -> str:
    """Merge a short follow-up with the last assistant turn from *history*.

    The returned string is used **only** for the retrieval query — the original
    *question* is still passed to the LLM prompt so the model answers what the
    user actually asked.

    Only the opening _REFORMULATE_TOPIC_SNIPPET_CHARS of the last assistant
    turn is used, not the full answer. This is the non-LLM fallback path —
    used when the primary _reformulate_query() call times out — so a long,
    detailed previous answer floods the retrieval query with unrelated detail
    and buries the follow-up's own content words. Confirmed live: after a
    multi-clause answer listing everything a policy document contains ("...the
    name and address of the insured, sum insured, period of insurance, risk
    covered, rate of premium, prescription of the subject matter..."), the
    follow-up "How can relatives be informed about the policy?" merged into a
    60+ word query where "relatives" was one word out of many — retrieval
    confidently matched the policy-document-CONTENTS topic (0.78 score)
    instead of anything about relatives. A short topic snippet still resolves
    pronouns against the established subject (the scenario this function
    exists for — see the call site's docstring) without drowning out a
    substantive follow-up's own words.
    """
    lines = history.strip().split("\n")
    last_assistant = ""
    for line in reversed(lines):
        if line.startswith("Assistant:"):
            last_assistant = line[len("Assistant:"):].strip()
            break
    if last_assistant:
        topic_snippet = last_assistant[:_REFORMULATE_TOPIC_SNIPPET_CHARS]
        return f"{topic_snippet} {question}"
    return question


def _is_insurance_related(question: str) -> bool:
    """Return True if *question* is plausibly about insurance or a related domain.

    Uses simple keyword/pattern matching — no LLM call.  Returns False for
    clearly off-topic queries (history, geography, tech comparisons, etc.).
    """
    q = question.lower().strip()

    # If the question is VERY short (1-2 words), assume it could be
    # insurance-related (e.g. "tell me more", "what about that") — avoids
    # false-off-topic for short follow-ups.  3-word queries like "chatgpt
    # vs claude" are long enough to carry clear topic content, so they
    # proceed to pattern matching below.
    if len(q.split()) <= 2:
        return True

    # ── Insurance / finance domain indicators ──────────────────────────────
    # Check these BEFORE off-topic patterns: if the query contains any actual
    # insurance vocabulary, it IS insurance-related regardless of phrasing.
    # This prevents phrases like "difference between term and whole life"
    # or "what is the history of insurance" from being falsely flagged as
    # off-topic just because they contain a generic pattern like "difference
    # between" or "history of".
    _INSURANCE_INDICATORS = re.compile(
        r"\b("
        r"insurance|policy|premium|deductible|coverage|claim|claims|"
        r"insure|insured|insurer|underwriting|underwrite|"
        r"cover|covered|covers|covers? (?:for|against|up to|of|on)|"
        r"premiums|co-pay|copay|deductibles|"
        r"health|medical|hospital|surgery|prescription|medication|"
        r"vehicle|car|motor|auto|bike|two-wheeler|four-wheeler|"
        r"travel|trip|flight|baggage|luggage|cancellation|"
        r"life|term life|whole life|endowment|ulip|"
        r"home|house|property|rental|landlord|tenant|"
        r"accident|disability|critical illness|cancer|"
        r"liability|third.party|comprehensive|"
        r"limit|limits|sum insured|sum.assured|"
        r"maternity|dental|vision|"
        r"agent|broker|renewal|grace period|waiting period|"
        r"no.claim|ncb|bonus|"
        r"nominee|beneficiary|"
        r"claim (?:form|process|settlement|rejection|approval)|"
        r"cashless|reimbursement|"
        r"roadside assistance|towing|"
        r"personal accident|"
        r"retirement|pension|annuity|"
        r"finance|financial|investment|savings|"
        r"hdfc ergo|icici|bajaj|tata aig|reliance|"
        r"new india|oriental|national|united india|"
        r"irda|regulator|"
        r"cover note|certi.* of insurance|"
        r"aog|marine|cargo|"
        r"group insurance|corporate|"
        r"rider|add.on"
        r")\b"
    )
    if _INSURANCE_INDICATORS.search(q):
        return True

    # ── Off-topic indicators ──────────────────────────────────────────────
    # These patterns only apply if the query has NO insurance vocabulary at
    # all.  Each sub-pattern uses \b word boundaries and \s+ for whitespace
    # between words (avoids trailing-space issues).
    _OFF_TOPIC_PATTERNS = re.compile(
        r"(?:"
        r"\b(?:who\s+is|who\s+was|who\s+are|when\s+was|when\s+did|when\s+is|where\s+is|where\s+was)\b|"
        r"\b(?:history\s+of|definition\s+of)\b|"
        r"\b(?:mother\s+of|father\s+of)\b|"
        r"\b(?:born\s+in|died\s+in|capital\s+of|population\s+of)\b|"
        r"\bchatgpt\b|\bclaude\b|\bgpt-4\b|"
        r"\brecipe\b|\bhow\s+to\s+cook\b|\bingredients\b|"
        r"\bhow\s+to\s+play\b|\brules\s+of\b|\bsoccer\b|\bfootball\b|\bcricket\b|\bbasketball\b|"
        r"\bmovie\b|\bactor\b|\bactress\b|\bsinger\b|\bsong\b|\balbum\b|"
        r"\bpython\b|\bjavascript\b|\bjava\b|\bc\+\+\b|\bprogramming\b|\bcode\b|\balgorithm\b|"
        r"\bweather\b|\btemperature\b|\bforecast\b|"
        r"\btranslate\b"
        r")"
    )
    if _OFF_TOPIC_PATTERNS.search(q):
        return False

    # For queries that don't match insurance indicators and don't match
    # off-topic patterns, fall back to generic question patterns combined
    # with insurance-adjacent words.

    # ── Generic question patterns (insurance-adjacent) ─────────────────────
    _GENERIC_INSURANCE_PATTERNS = re.compile(
        r"\b("
        r"what (?:are|is|does|about)|"
        r"how (?:much|many|does|can|to|do)|"
        r"can (?:i|we|you)|"
        r"do (?:i|we|you)|"
        r"tell me about|explain|"
        r"benefits|features|details|"
        r"am i|is it|will it|"
        r"recommend|suggest|"
        r"best (?:for|option|plan|policy|)"
        r")\b"
    )
    # Only count generic patterns if they contain at least one insurance-adjacent word
    _INSURANCE_ADJACENT = re.compile(
        r"\b(cover|protect|risk|plan|option|policy|benefit|pay|cost|fee|charge|"
        r"amount|document|upload|file|paper|letter|receipt)"
        r"\b"
    )
    if _GENERIC_INSURANCE_PATTERNS.search(q) and _INSURANCE_ADJACENT.search(q):
        return True

    # Default: when in doubt, assume it IS insurance-related (better to let the
    # retrieval similarity decide than to falsely mark as off-topic).
    return True


from langchain_core.documents import Document
from rag import RAGPipeline
from router import get_insurance_llm
from video_store import VideoVectorStore
from webpage_store import WebpageVectorStore
from calculator import compute_insurance_benefits, _is_calculation_question
from prompt_template import (
    STRICT_GROUNDED_PROMPT, DETAILED_GROUNDED_PROMPT,
    CALCULATION_PROMPT, CONVERSATIONAL_RAG_PROMPT,
)
from context_compressor import ContextCompressor
from rag import LLM_CONTEXT_WINDOW_CHARS

# ask_stream()'s dynamic context budget (below) used to reserve a flat,
# hardcoded 700 tokens for "prompt template boilerplate" regardless of
# which of the three prompts actually gets used. That number went stale
# the moment the tone/warmth rewrite grew all three templates well past it
# (measured: STRICT_GROUNDED_PROMPT ~1205, DETAILED_GROUNDED_PROMPT
# ~1234, CONVERSATIONAL_RAG_PROMPT ~1889 tokens) — confirmed live: a real
# user's "explain point 4 with an example" follow-up in an active session
# with real accumulated history hit the model's 4096-token ceiling and
# crashed with "internal error", because the budget calculation thought
# it only needed to reserve 700 tokens for the template it was about to
# use and left too much room for history + context. Computed here from
# the actual prompt strings (not hardcoded) so this can't silently go
# stale again the next time any of the three templates changes size.
#
# _CHARS_PER_TOKEN=4 was the original assumption here (a common rule of
# thumb for English text), but confirmed live it undercounts real usage
# by ~30-35% for this prompt's actual mix of insurance jargon, source
# citations, and punctuation-heavy formatting: a request that this math
# estimated as safely within budget was rejected by the real backend at
# 3585 real input tokens + 512 requested output = 4097, just over the
# model's 4096-token ceiling. Using 3 chars/token (instead of 4) builds
# in that margin everywhere this ratio is used for a token estimate.
_CHARS_PER_TOKEN = 3
_STRICT_PROMPT_TOKENS_EST = len(STRICT_GROUNDED_PROMPT) // _CHARS_PER_TOKEN
_DETAILED_PROMPT_TOKENS_EST = len(DETAILED_GROUNDED_PROMPT) // _CHARS_PER_TOKEN
_CONVERSATIONAL_PROMPT_TOKENS_EST = len(CONVERSATIONAL_RAG_PROMPT) // _CHARS_PER_TOKEN

class MultiSourceRAG:
    def __init__(self, doc_pipeline: Optional[RAGPipeline] = None):
        # Accepts an existing RAGPipeline rather than always constructing a
        # fresh one — critical when the caller already has a live singleton
        # (see api.py's _get_multi_rag(), which passes _get_pipeline()'s
        # result). Two independently-constructed RAGPipeline instances each
        # load their own in-memory copy of the same on-disk vector store and
        # KV cache; a document inserted via one is invisible to queries
        # served through the other until a process restart reloads both
        # from disk. Confirmed live: /upload (which used the module-level
        # _get_pipeline() singleton) was invisible to /ask-stream (which
        # used a separately-constructed MultiSourceRAG().doc_pipeline) for
        # the lifetime of the process — a document could be uploaded
        # successfully and still get "not in my knowledge base" on every
        # query until the container restarted, regardless of any cache
        # state. /upload-webpage and /upload-video were unaffected only
        # because they already happened to insert through the
        # MultiSourceRAG instance directly.
        self.doc_pipeline = doc_pipeline if doc_pipeline is not None else RAGPipeline()
        self.video_store = VideoVectorStore()
        self.webpage_store = WebpageVectorStore()
        self.max_context_chars = LLM_CONTEXT_WINDOW_CHARS  # kept in sync with compress_to_budget budget
        # Share the embed model already loaded by doc_pipeline — no duplicate memory.
        self._compressor = ContextCompressor(
            embed_model=self.doc_pipeline.vector_store.embed_model,
            similarity_threshold=0.38,
            min_sentences=2,
            max_sentences=10,
            max_chars_per_chunk=LLM_CONTEXT_WINDOW_CHARS,
        )

    def _merge_chunks(self, chunks: List[Document]) -> List[Document]:
        seen = {}
        for chunk in chunks:
            h = hash(chunk.page_content[:200])
            if h not in seen:
                seen[h] = chunk
            else:
                # Prefer the version with the higher relevance score.
                # rerank_score (from BGE cross-encoder) is more informative than
                # the raw retrieval similarity, so use it when available.
                def _best_score(d):
                    return d.metadata.get("rerank_score", d.metadata.get("similarity", 0))
                if _best_score(chunk) > _best_score(seen[h]):
                    seen[h] = chunk
        return list(seen.values())

    async def _retrieve_all_sources_combined(
        self,
        retrieval_query: str,
        filter_meta: Optional[dict],
        doc_top_k: int,
        summary_top_k: int,
        media_top_k: int,
        chunk_limit: int,
    ) -> List[Document]:
        """
        Fetch raw (unreranked) candidates from doc, video, and webpage
        stores, merge them, and rerank ONCE across the combined pool —
        instead of the previous approach of reranking each source
        separately (doc via rerank_documents, video/webpage each via
        their own search(use_reranker=True) call).

        doc, video, and webpage all share ONE process-wide CrossEncoder
        instance (_get_shared_reranker in turbovec_store.py), so three
        separate reranking calls each pay their own fixed per-call
        overhead on this deployment's CPU regardless of candidate count
        — confirmed live: an isolated warm-steady-state video
        search+rerank call alone measured ~7s on every single call, for
        a 28-chunk store that showed up in a final answer's citations in
        roughly 1 of 30+ test questions that day. Reranking everything
        together pays that fixed cost once.

        This also directly implements "only retrieve what actually
        supports the answer, not a fixed count per source": a source
        with nothing genuinely relevant to this query contributes zero
        chunks to the final answer on its own merits — its candidates
        simply score too low against the rest of the combined pool in
        the SAME ranking — rather than doc/video/webpage each always
        claiming their own fixed number of slots in the final context
        regardless of whether anything in a given source is actually
        useful for this specific question.
        """
        doc_raw = await asyncio.to_thread(
            self.doc_pipeline._vector_store.search,
            retrieval_query, top_k=doc_top_k, use_hybrid=True, use_reranker=False,
            filter_metadata=filter_meta,
        )

        # Skip a store entirely when it's empty — search() would return []
        # almost instantly anyway (count()==0 early return), but this also
        # means an empty store never even reaches the asyncio.to_thread
        # dispatch, and stays skipped automatically once the store has
        # exactly zero content, without ever needing separate code to
        # re-enable it if content gets added later.
        video_raw: List[Document] = []
        if self.video_store.count() > 0:
            video_raw = await asyncio.to_thread(
                self.video_store.search, retrieval_query, top_k=media_top_k, use_hybrid=True, use_reranker=False,
            )
        webpage_raw: List[Document] = []
        if self.webpage_store.count() > 0:
            webpage_raw = await asyncio.to_thread(
                self.webpage_store.search, retrieval_query, top_k=media_top_k, use_hybrid=True, use_reranker=False,
            )

        # Stage-1 summary-boost guarantee (same logic as _retrieve_doc_chunks):
        # a summary-identified document's best-matching chunk may not have
        # ranked highly enough in the raw top-k search to be included, so
        # fetch its top 2 chunks directly and fold them into the combined
        # pool — unreranked, they get scored fairly alongside everything
        # else in the single rerank call below.
        if self.doc_pipeline._summary_store.count() > 0:
            try:
                relevant_summaries = await asyncio.to_thread(
                    self.doc_pipeline._summary_store.search, retrieval_query, summary_top_k
                )
                seen_summary_srcs: set = set()
                for summary_doc in relevant_summaries:
                    src = summary_doc.metadata.get("source", "")
                    if not src or src in seen_summary_srcs:
                        continue
                    seen_summary_srcs.add(src)
                    # Respect the same policy_type constraint the main search
                    # already applies — a document's SUMMARY can score well
                    # against a query on totally different, coincidentally-
                    # similar-sounding grounds (e.g. any "how do I file a
                    # claim" summary tends to resemble any other), and unlike
                    # the main search this boost has no type check of its
                    # own. Confirmed live: a newly-uploaded pet insurance
                    # guide's summary matched "how to claim health insurance
                    # for a fractured hand" closely enough to enter the top
                    # summary matches, and its claims-process chunk (source-
                    # filtered only, no type filter) got boosted straight
                    # into a human health-insurance answer verbatim ("Most
                    # insurers accept claims through a mobile app... photo
                    # submission"). filter_meta already carries "$in":
                    # [_query_policy_type, "general"] when the query
                    # classified to a specific type — folding it in here
                    # closes that gap without touching the boost's own
                    # purpose (rescuing an under-ranked chunk from a
                    # genuinely relevant, type-compatible document).
                    _boost_filter = (
                        {"$and": [filter_meta, {"source": {"$eq": src}}]}
                        if filter_meta else {"source": {"$eq": src}}
                    )
                    boost = await asyncio.to_thread(
                        self.doc_pipeline._vector_store.search,
                        retrieval_query, 2, _boost_filter, True, False,
                    )
                    if boost:
                        existing = {d.page_content[:80] for d in doc_raw}
                        new_boost = [d for d in boost if d.page_content[:80] not in existing]
                        for d in new_boost:
                            d.metadata["stage1_boost"] = True
                        doc_raw = doc_raw + new_boost
            except Exception as _exc:
                logger.debug("[MultiSourceRAG] stage-1 guarantee skipped: %s", _exc)

        combined = self._merge_chunks(doc_raw + video_raw + webpage_raw)
        if not combined:
            return []

        return await asyncio.to_thread(
            self.doc_pipeline._vector_store.rerank_documents,
            retrieval_query, combined, chunk_limit,
        )

    async def _retrieve_doc_chunks(
        self,
        retrieval_query: str,
        filter_meta: Optional[dict],
        document_filter: Optional[List[str]],
        doc_top_k: int = 30,
        summary_top_k: int = 5,
        rerank_top_k: int = 8,
    ) -> List[Document]:
        """Run doc-vector search, stage-1 summary-boost loop, and rerank.

        Returns the final reranked list of document chunks.
        The summary-boost loop (stage-1 source guarantee) only runs when
        *document_filter* is falsy and the SummaryStore is non-empty.
        """
        doc_chunks = await asyncio.to_thread(
            self.doc_pipeline._vector_store.search,
            retrieval_query, top_k=doc_top_k, use_hybrid=True, use_reranker=False,
            filter_metadata=filter_meta
        )

        if not document_filter and self.doc_pipeline._summary_store.count() > 0:
            try:
                relevant_summaries = await asyncio.to_thread(
                    self.doc_pipeline._summary_store.search, retrieval_query, summary_top_k
                )
                seen_summary_srcs: set = set()
                for summary_doc in relevant_summaries:
                    src = summary_doc.metadata.get("source", "")
                    if not src or src in seen_summary_srcs:
                        continue
                    seen_summary_srcs.add(src)
                    # Always boost from summary-identified documents even if
                    # they are already partially in the pool.  The initial top-k
                    # may have fetched the wrong sections of that document; the
                    # boost fetches the 2 chunks most relevant to THIS query.
                    # (Was silently 5 for a while, paired with summary_top_k=3
                    # at the call sites — up to 15 extra candidates before
                    # reranking even started, a major, disproportionate driver
                    # of reranking latency for a modest recall benefit. Capped
                    # back down to what the comment always said, and
                    # summary_top_k trimmed to 2 alongside it — now at most
                    # 2 docs x 2 chunks = 4 extra candidates.)
                    # Same policy_type guard as _retrieve_all_sources_combined's
                    # copy of this loop — see the comment there for the
                    # confirmed-live cross-topic leak this closes.
                    _boost_filter = (
                        {"$and": [filter_meta, {"source": {"$eq": src}}]}
                        if filter_meta else {"source": {"$eq": src}}
                    )
                    boost = await asyncio.to_thread(
                        self.doc_pipeline._vector_store.search,
                        retrieval_query, 2, _boost_filter, True, False,
                    )
                    if boost:
                        existing = {d.page_content[:80] for d in doc_chunks}
                        new_boost = [
                            d for d in boost if d.page_content[:80] not in existing
                        ]
                        for d in new_boost:
                            d.metadata["stage1_boost"] = True
                        doc_chunks = doc_chunks + new_boost
                        if new_boost:
                            logger.info(
                                "[MultiSourceRAG] stage-1 boost: added %d chunk(s) from %r",
                                len(new_boost), src,
                            )
            except Exception as _exc:
                logger.debug("[MultiSourceRAG] stage-1 guarantee skipped: %s", _exc)

        if doc_chunks:
            doc_chunks = await asyncio.to_thread(
                self.doc_pipeline._vector_store.rerank_documents,
                retrieval_query, doc_chunks, rerank_top_k,
            )

        return doc_chunks

    async def ask(self, question: str, history: str = "", document_filter: Optional[List[str]] = None) -> Tuple[str, List[str], bool, bool]:
        """
        Returns (answer, sources, needs_human, is_off_topic).
        needs_human is True when no relevant context was found — meaning the
        model has no grounding and is answering from general knowledge or not
        at all.  The caller should flag the query for human follow-up.
        is_off_topic is True when the question is clearly not insurance-related
        (e.g. history, geography, tech comparisons) — the caller should return
        a friendly refusal instead of a human handoff.
        """
        # ── Short follow-up reformulation ──────────────────────────────────────
        # Very short messages ("yes", "ok", "what about that") have no standalone
        # semantic content, so raw retrieval returns near-zero similarity and
        # incorrectly triggers human handoff.  If history exists, reformulate the
        # retrieval query by merging with the last assistant turn.
        is_short = _is_short_followup(question)
        has_history = bool(history.strip())
        retrieval_query = _reformulate_with_history(question, history) if (is_short and has_history) else question

        # ── LLM-based query contextualization (replaces keyword follow-up detection) ──
        # Resolves pronouns and implicit references against recent conversation
        # history on every turn. The rewritten query is used for retrieval and
        # coverage checks; the original question is kept for the LLM prompt so
        # the model answers what the user actually asked.
        _contextualized = await _contextualize_query(retrieval_query, history)
        if _contextualized != retrieval_query:
            # Also log what the old keyword-based classifier would have said,
            # for side-by-side comparison in production logs before removal.
            _old_followup = _is_likely_followup(question) if history else False
            logger.info(
                "[CTX] %r → %r (old _is_likely_followup=%s)",
                retrieval_query, _contextualized, _old_followup,
            )
            retrieval_query = _contextualized

        # Build filter
        filter_meta = None
        if document_filter:
            conditions = [{"source": {"$contains": doc}} for doc in document_filter]
            filter_meta = conditions[0] if len(conditions) == 1 else {"$or": conditions}
            logger.info(f"Document filter: {document_filter}")

        # ── Parallel retrieval across all sources ─────────────────────────────
        if not document_filter:
            doc_chunks, video_chunks, webpage_chunks = await asyncio.gather(
                self._retrieve_doc_chunks(retrieval_query, filter_meta, document_filter),
                asyncio.to_thread(self.video_store.search, retrieval_query, top_k=4, use_hybrid=True, use_reranker=True),
                asyncio.to_thread(self.webpage_store.search, retrieval_query, top_k=4, use_hybrid=True, use_reranker=True),
            )
            all_chunks = self._merge_chunks(doc_chunks + video_chunks + webpage_chunks)
        else:
            doc_chunks = await self._retrieve_doc_chunks(retrieval_query, filter_meta, document_filter)
            all_chunks = self._merge_chunks(doc_chunks)

        all_chunks.sort(key=lambda x: x.metadata.get("similarity", 0), reverse=True)
        all_chunks = all_chunks[:8]

        # --- Determine whether retrieved content is relevant enough to ground the answer ---
        # If the top chunk has similarity <= 0.05, the retrieval found nothing
        # meaningfully relevant. Flag needs_human so the caller can trigger a human handoff.
        # 
        # Short follow-ups with available history bypass raw-similarity detection:
        # the reformulated query should retrieve meaningful context, and even if
        # scores are low the conversation history provides enough grounding.
        top_similarity = all_chunks[0].metadata.get("similarity", 0) if all_chunks else 0
        needs_human = (top_similarity <= 0.05)
        if is_short and has_history:
            needs_human = False

        # ── Off-topic detection ────────────────────────────────────────────────
        # If retrieval found nothing relevant AND the question is clearly not
        # about insurance, mark it as off-topic so the caller can give a friendly
        # refusal instead of triggering a human handoff.
        is_off_topic = False
        if needs_human and not _is_insurance_related(question):
            is_off_topic = True
            needs_human = False
            # Skip the LLM entirely — return a firm, friendly refusal
            return (
                "I'm Layla, your insurance assistant! I can only help with insurance-related questions, things like policy coverage, premiums, claims, and benefits. Is there something about your insurance I can help you with today? 😊",
                [],
                False,
                True,
            )
        # ── Context compression (only when needed) ────────────────────────────
        # Skip compression entirely when the chunks already fit in the LLM's
        # input window — with 500-char chunks this will usually be the case.
        # Only compress when the aggregate exceeds LLM_CONTEXT_WINDOW_CHARS.
        total_retrieved_chars = sum(len(c.page_content) for c in all_chunks)
        if total_retrieved_chars > LLM_CONTEXT_WINDOW_CHARS:
            logger.info(
                "[MultiSourceRAG] Context too large (%d chars > %d limit) — compressing",
                total_retrieved_chars, LLM_CONTEXT_WINDOW_CHARS,
            )
            all_chunks = self._compressor.compress_to_budget(
                question, all_chunks, max_total_chars=LLM_CONTEXT_WINDOW_CHARS
            )

        # Build context
        _VIDEO_SOURCE_TYPES = {"video", "youtube_transcript", "youtube"}
        _WEBPAGE_SOURCE_TYPES = {"webpage", "web"}
        context_parts, sources = [], []
        for chunk in all_chunks:
            source_type = chunk.metadata.get("source_type", "document")
            doc_type = chunk.metadata.get("doc_type", "")
            if source_type in _VIDEO_SOURCE_TYPES or doc_type == "youtube":
                url = chunk.metadata.get("source_url") or chunk.metadata.get("source", "Unknown URL")
                title = chunk.metadata.get("video_title", "")
                label = f"Video: {title or url}"
                sources.append(url)
            elif source_type in _WEBPAGE_SOURCE_TYPES:
                url = chunk.metadata.get("source_url") or chunk.metadata.get("source", "Unknown URL")
                label = f"Webpage: {url}"
                sources.append(url)
            else:
                src = chunk.metadata.get("source", "Unknown")
                page = chunk.metadata.get("page", "?")
                label = f"Document: {src} (Page {page})"
                sources.append(f"{src} (page {page})")
            context_parts.append(f"[{label}]\n{chunk.page_content}")
        full_context = "\n\n".join(context_parts)
        if len(full_context) > self.max_context_chars:
            full_context = full_context[:self.max_context_chars] + "... (truncated)"

        # Calculation
        calc_answer, is_calc = compute_insurance_benefits(question, full_context)
        if is_calc or _is_calculation_question(question):
            prompt = CALCULATION_PROMPT.format(
                context=full_context or "No relevant content found.",
                history=history,
                question=question
            )
            llm = get_insurance_llm(temperature=0)
            try:
                response = await asyncio.to_thread(llm.invoke, prompt)
                answer = response.content if hasattr(response, "content") else str(response)
            except _LLM_BACKEND_ERRORS as _exc:
                logger.warning("[MultiSourceRAG] LLM backend unavailable for calculation: %s", _exc)
                answer = "I'm sorry, I can't process that calculation right now. The AI model server seems to be unreachable. Please try again in a moment!"
            return _strip_markdown(_strip_model_preamble(answer)), list(dict.fromkeys(sources)), needs_human, is_off_topic

        if not full_context.strip():
            # No documents retrieved — return a firm refusal rather than letting
            # the LLM answer from training knowledge (small 7B models ignore grounding
            # instructions when context is empty).
            return (
                "Hmm, I don't have that specific information in my knowledge base right now. "
                "Let me get one of our agents on it, they'll be able to help you better! 😊",
                [],
                True,
                False,
            )

        # ── Prompt selection ──────────────────────────────────────────────────
        # Lexical checks run alongside the semantic _verify_grounding() backstop
        # via asyncio.gather() rather than after it, so the LLM round-trip isn't
        # serialized behind work that's already fast. Both must pass.
        # Both the lexical checks and _verify_grounding() receive the
        # contextualized retrieval_query (not the original question) so that
        # pronoun-resolved terms like "takaful principles" are checked against
        # the same query that was actually used for retrieval, not the raw
        # unresolved question containing "it's".
        async def _lexical_covered():
            return (
                _context_covers_query(retrieval_query, all_chunks)
                and _quoted_comparison_covered(retrieval_query, all_chunks)
                and _enumeration_query_covered(retrieval_query, all_chunks)
            )
        _lex_ok, _semantically_grounded = await asyncio.gather(
            _lexical_covered(),
            _verify_grounding(retrieval_query, full_context),
        )
        ctx_covered = _lex_ok and _semantically_grounded

        # If the KB has no relevant content for this question, skip the LLM
        # entirely — small models ignore "don't use your training knowledge"
        # instructions and answer from general knowledge anyway. Return a hard
        # canned response so the handoff trigger fires reliably.
        if not ctx_covered and not document_filter:
            return (
                "Hmm, I don't have that specific information in my knowledge base right now. "
                "Let me get one of our agents on it, they'll be able to help you better! 😊",
                [],
                needs_human,
                is_off_topic,
            )

        if document_filter:
            prompt = STRICT_GROUNDED_PROMPT.format(history=history, context=full_context, question=question)
            llm = get_insurance_llm(temperature=0)
        else:
            prompt = CONVERSATIONAL_RAG_PROMPT.format(
                history=history,
                context=full_context,
                question=question,
            )
            llm = get_insurance_llm(temperature=0)

        # ── LLM invocation with backend-error guard ───────────────────────────
        # When the context does NOT cover the query (out-of-KB question) and the
        # LLM backend is unreachable, provide an immediate graceful fallback
        # rather than waiting for a 150-second timeout.
        try:
            response = await asyncio.to_thread(llm.invoke, prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except _LLM_BACKEND_ERRORS as _exc:
            logger.warning(
                "[MultiSourceRAG] LLM backend unavailable (ctx_covered=%s): %s",
                ctx_covered, _exc,
            )
            if not ctx_covered:
                # Out-of-KB query + backend down: inform user their question is valid
                # but not in the KB AND the LLM is temporarily unreachable.
                answer = (
                    "Hmm, that topic doesn't seem to be covered in my knowledge base right now. "
                    "On top of that, my AI model server is temporarily unreachable, so I can't "
                    "pull from general knowledge either. Try again in a moment, or feel free to "
                    "ask me something about your uploaded insurance documents!"
                )
            else:
                answer = (
                    "I couldn't reach the AI model server to generate your answer right now. "
                    "Please try again in a moment!"
                )
        answer = _strip_markdown(_strip_model_preamble(answer))
        return answer, list(dict.fromkeys(sources)), needs_human, is_off_topic

    async def ask_stream(
        self,
        question: str,
        history: str = "",
        document_filter: Optional[List[str]] = None,
    ):
        """Async generator — yields text tokens as the LLM produces them.

        Runs all retrieval logic identically to ask(), then streams the LLM
        response token-by-token so the frontend can show words appearing live
        instead of waiting for the full answer.

        Yields:
            str tokens as they arrive, then a final JSON line:
            'data: {"sources": [...], "done": true}'
        """
        # ── Re-use the full retrieval pipeline ───────────────────────────────
        # Build the prompt exactly as ask() does, then stream the LLM response.
        # We call ask() with a sentinel and intercept just before llm.invoke.
        # Simpler: duplicate the prompt-building block here (it's fast, <1s).

        # Per-phase latency tracking (2026-07-10, explicit user request) — a
        # single consolidated log line at the end of the main generation
        # path breaking down where the request's wall-clock time actually
        # went (retrieval, grounding checks, LLM generation), so a slow
        # request can be diagnosed from logs alone instead of guessing.
        # Left as None on the fast early-return paths
        # (cache hit, refusal) since those aren't the latency problem this
        # was asked for — they're already fast, and instrumenting every one
        # of this function's many early returns would add a lot of noise
        # for no diagnostic value.
        _t_request_start = time.time()
        _t_retrieval_ms = None
        _t_grounding_ms = None
        _t_llm_ms = None
        _t_preprocess_ms = None
        _t_promptbuild_ms = None
        _t_postllm_ms = None
        # Time-to-first-token — set once, on the first real generated token
        # actually yielded to the client. Distinct from _t_llm_ms (which
        # times the WHOLE generation call): this is what the user actually
        # perceives as "how long until something starts appearing," i.e.
        # preprocess+retrieval+grounding+promptbuild+(prefill) — the exact
        # "dead air" the latency plan (plan_latency.md) Phase 2 targets,
        # separate from total generation time (Phase 1's GPU-vLLM lever).
        _t_ttft_ms = None

        retrieval_query = question
        filter_meta = None
        if document_filter:
            conditions = [{"source": {"$contains": doc}} for doc in document_filter]
            filter_meta = conditions[0] if len(conditions) == 1 else {"$or": conditions}

        # ── Last assistant turn (for meta-conversation and handoff checks) ─────
        # Compute once before any fast-path check so Parts 3 and 4 can reuse it.
        _last_assistant_turn = ""
        if history:
            for _turn in reversed(_split_history_turns(history)):
                if _turn.startswith("Assistant:"):
                    _last_assistant_turn = _turn[len("Assistant:"):].strip()
                    break
        _last_was_refusal_or_error = any(
            p in _last_assistant_turn.lower() for p in (
                "i don't have that specific information",
                "i don't have that in my knowledge base",
                "let me get one of our agents",
                "let me get a human agent",
                "couldn't reach the ai model server",
                "could not generate an answer",
                "taking too long",
            )
        )

        # ── Pure conversational replies — no retrieval needed ─────────────────
        # "yes", "no", "ok", "thanks" etc. have zero retrieval value.
        _PURE_CONV = frozenset({
            "yes", "no", "ok", "okay", "sure", "alright", "nope", "nah",
            "thanks", "thank you", "got it", "i see", "understood", "right",
            "cool", "great", "nice", "fine", "good", "perfect", "awesome",
            "no thanks", "no thank you", "not really", "never mind", "nevermind",
        })
        _HANDOFF_AFFIRM = frozenset({"yes", "sure", "ok", "okay", "yeah", "yep", "please", "yes please"})
        _q_stripped = question.strip().lower().strip("!.,?;:")
        if _q_stripped in _PURE_CONV:
            # ── Handoff-affirm fast path (Part 4) ─────────────────────────
            # If the last assistant message was a handoff offer and the user
            # says "yes"/"sure"/etc., acknowledge the handoff request with
            # needs_human=True so api.py's existing handoff-trigger logic fires.
            _wants_handoff = (
                _q_stripped in _HANDOFF_AFFIRM
                and any(p in _last_assistant_turn.lower() for p in (
                    "connect you with one",
                    "let me get one of our agents",
                    "let me get a human agent",
                ))
            )
            if _wants_handoff:
                import json as _json_s
                yield "Sure thing! Connecting you with a human agent now, one moment. 😊"
                yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": True})
                return
            if _q_stripped in {"yes", "sure", "ok", "okay", "alright", "cool", "great", "perfect", "awesome"}:
                conv_reply = "Great! Let me know if you have any other questions about insurance. 😊"
            elif _q_stripped in {"no", "nope", "nah", "no thanks", "no thank you", "not really"}:
                conv_reply = "No problem! Feel free to ask me anything else about your insurance. 😊"
            elif _q_stripped in {"thanks", "thank you"}:
                conv_reply = "You're welcome! Let me know if there's anything else I can help you with. 😊"
            else:
                conv_reply = "Sure! Let me know if you have any other questions. 😊"
            import json as _json_s
            yield conv_reply
            yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": False})
            return

        # ── Identity / capability questions — no retrieval needed ─────────────
        # CONVERSATIONAL_RAG_PROMPT has an IDENTITY RULES section with the
        # canned answers below, but that's useless if the request never
        # reaches the LLM at all — confirmed live: "Who built you?" has zero
        # insurance vocabulary, so retrieval found nothing, ctx_covered was
        # False, and it fell straight through to the "I don't have that in my
        # knowledge base" refusal + an unnecessary human-escalation email,
        # never once consulting the prompt that already knew how to answer
        # it. Same lesson as the off-topic fast-path and
        # _is_conversational_reaction fixes: short-circuit deterministically
        # before retrieval for questions that structurally can't be answered
        # by KB lookup, rather than letting them reach the refusal path by
        # default. Text matches the prompt's own IDENTITY RULES wording so
        # the two stay in sync.
        # "are you ... " permissive of 0-3 filler words before the actual
        # identity term (real/human/bot/AI/person) — confirmed live the
        # exact-adjacency version missed "are you a real person or a bot"
        # entirely (neither alternative sits directly after "are you" once
        # the other option is mentioned first), same fragility class as
        # [[project_detail_pattern_generalization]]'s "in detail" gap.
        _IDENTITY_RE = re.compile(
            r"\b(who (?:built|made|created|developed|designed) you|"
            r"who(?:'s| is) behind you|"
            r"who do you work for|"
            r"what company (?:built|made|created|owns) you|"
            r"what are you built (?:on|with)|"
            r"are you (?:\w+\s+){0,3}?(?:chatgpt|gpt|an? ai|a bot|a real (?:person|human)|"
            r"a human|human|real|a person)|"
            r"is this (?:a bot|an? ai|chatgpt|gpt)|"
            r"am i (?:talking|chatting|speaking) (?:to|with) (?:a bot|an? ai|a human|a real person)|"
            r"what model (?:are you|is this))\b",
            re.IGNORECASE,
        )
        _KNOWLEDGE_SCOPE_RE = re.compile(
            r"\b(what do you know|what can you (?:help|do)|"
            r"what (?:kind|type) of (?:questions|things) can you|"
            r"what topics (?:do you|can you)|"
            r"what are you (?:trained|good) (?:on|at))\b",
            re.IGNORECASE,
        )
        if _IDENTITY_RE.search(question):
            import json as _json_s
            yield (
                "I was built by Nexsys IT Consulting, a tech firm that builds smart AI solutions. "
                "Pretty cool, right? 😊 Anyway, I'm here for you. What insurance question can I help with?"
            )
            yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": False})
            return
        if _KNOWLEDGE_SCOPE_RE.search(question):
            import json as _json_s
            yield "I've got a lot of insurance knowledge: health, life, motor, travel, home and more. What's on your mind?"
            yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": False})
            return

        # ── Meta-conversation clarification after refusal/error (Part 3) ──────
        # If the previous assistant turn was a refusal/error/handoff message
        # and the user's very next message is a short clarifying question about
        # THAT message itself ("meaning?", "why?", "what does that mean"), answer
        # conversationally instead of attempting KB retrieval on it.
        _META_CLARIFY_PHRASES = frozenset({
            "meaning", "meaning?", "what does that mean", "what does this mean",
            "what do you mean", "why", "why not", "why is that", "what happened",
            "what does that mean?", "i don't understand", "i dont understand",
            "huh", "huh?", "what",
        })
        _q_meta_check = question.strip().lower().strip("!.,?;:")
        if _last_was_refusal_or_error and _q_meta_check in _META_CLARIFY_PHRASES:
            import json as _json_s
            yield (
                "Sorry about that! I meant I couldn't find specific details on that "
                "topic in my knowledge base right now, so I wanted to loop in a human "
                "agent who could help further. Want me to connect you with one, or "
                "would you like to try asking in a different way?"
            )
            yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": False})
            return

        # ── Casual greeting / small talk — fuzzy match ────────────────────────
        # If the stripped/lowercased question is <= 3 words, fuzzy match against
        # _CASUAL_GREETINGS.  If score >= 75, treat as greeting and skip retrieval.
        _q_lower = question.strip().lower()
        _q_words = _q_lower.split()
        if len(_q_words) <= 3:
            _greeting_result = process.extractOne(_q_lower, _CASUAL_GREETINGS, scorer=fuzz.ratio)
            if _greeting_result is not None and _greeting_result[1] >= 75:
                import json as _json_s
                yield "Hey there! 👋 I'm Layla, your insurance assistant. How can I help you today?"
                yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": False})
                return

        # ── User-statement fast path ──────────────────────────────────────────
        # "I have a health plan", "I got term insurance last month" → acknowledge warmly.
        _stmt = re.match(
            r"^\s*i\s+(have|got|have\s+got|purchased|bought|own|took|taken|recently\s+got|just\s+got"
            r"|am\s+covered|am\s+insured|enrolled|signed\s+up)\b",
            question.lower(),
        )
        if _stmt and _is_insurance_related(question):
            _plan_word = next(
                (w for w in ("health", "life", "motor", "car", "travel", "home", "term", "ulip", "vehicle")
                 if w in question.lower()), "insurance"
            )
            import json as _json_s
            yield (
                f"That's great that you have a {_plan_word} plan! "
                f"I'm here to help you understand it better. "
                f"What would you like to know about your coverage, claims, premiums, or anything else?"
            )
            yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": False})
            return

        # ── Pasted-context follow-up detection ────────────────────────────────
        # A user pasting a chunk of an earlier answer (possibly from many
        # turns back, well outside the retained history window) plus a
        # question about it should be answered using that pasted text as
        # context — not treated as one long, garbled retrieval query. From
        # here on, `question` is the real, short question being asked;
        # `pasted_context` (if any) is folded into full_context further down,
        # and `_pasted_grounds_answer` lets the refusal gates below know the
        # paste alone already supports an answer, independent of whatever
        # fresh KB retrieval turns up.
        pasted_context, question = await _extract_pasted_followup(question)
        _pasted_is_point_reference = False
        if pasted_context is None:
            # No literal paste in this message — but a short "explain point 2"
            # style follow-up references a numbered list from the PREVIOUS
            # answer, which is already sitting in history. Pull that point's
            # own text out and feed it through the same path as a live paste,
            # rather than depending on retrieval rediscovering it from the KB
            # (unreliable — see _extract_point_text_from_history's docstring).
            _point_num = _extract_point_number(question)
            if _point_num is not None and history:
                pasted_context = _extract_point_text_from_history(history, _point_num)
                _pasted_is_point_reference = pasted_context is not None
        _pasted_grounds_answer = False
        if pasted_context:
            _pasted_grounds_answer = await _verify_grounding(question, pasted_context)
        # For a point-reference follow-up specifically, fetch supplementary
        # KB content using the POINT'S OWN TEXT as the retrieval query, not
        # the vague "explain point 3 in more detail" phrasing the main
        # pipeline's retrieval further down would otherwise use. Confirmed
        # live this distinction matters, not just as a precaution: with the
        # main pipeline's generically-retrieved content merged in instead
        # of this, "explain point 3" (about machinery breakdown / boiler /
        # electronic equipment risks) came back answering about the
        # indemnity principle instead — an entirely different KB chunk that
        # happened to rank for the vague follow-up text, silently
        # overriding what the user actually asked about. Retrieving with
        # the point's own subject matter as the query keeps whatever
        # supplementary content is found on-topic.
        _point_ref_context = ""
        _point_ref_sources: list = []
        if _pasted_is_point_reference and pasted_context:
            try:
                _point_ref_chunks = await self._retrieve_doc_chunks(
                    pasted_context, filter_meta, document_filter, doc_top_k=6, summary_top_k=2,
                )
                # Retrieving with the point's own text as the query narrows
                # the candidate pool, but embedding similarity alone can
                # still rank a DIFFERENT, merely topically-adjacent policy
                # type highly (both live under the same "engineering
                # insurance" umbrella) — confirmed live: point 3 was about
                # the project-stage/procurement risks, but a chunk about
                # Business Interruption Insurance (no real overlap in
                # subject matter, just the same broad section of the KB)
                # still made it through and the model answered about THAT
                # instead. Re-use the same significant-word-overlap check
                # the padding filter uses (project_padding_filter_over_
                # aggressive.md) to keep only chunks that actually share
                # vocabulary with the point's own text, not just its
                # neighborhood in embedding space.
                _point_words = {
                    w for w in re.findall(r"\w+", pasted_context.lower()) if len(w) >= 5
                }
                def _shares_vocabulary(text: str) -> bool:
                    if not _point_words:
                        return True
                    chunk_words = {w for w in re.findall(r"\w+", text.lower()) if len(w) >= 5}
                    return len(_point_words & chunk_words) >= min(3, max(1, len(_point_words) // 4))
                _point_ref_chunks = [
                    c for c in _point_ref_chunks
                    if _shares_vocabulary(getattr(c, "page_content", ""))
                ]
                if _point_ref_chunks:
                    _point_ref_context = "\n\n".join(
                        c.page_content for c in _point_ref_chunks if getattr(c, "page_content", "")
                    )[:4000]
                    # Citation accuracy: these are the chunks the answer will
                    # actually be grounded in for this branch, not whatever
                    # the main pipeline's (mistargeted, for this case)
                    # retrieval further down finds.
                    for c in _point_ref_chunks:
                        src = c.metadata.get("source", "Unknown")
                        page = c.metadata.get("page", "?")
                        _point_ref_sources.append(f"{src} (page {page})")
            except Exception:
                pass

        # ── Compound-question disambiguation ───────────────────────────────────
        # A generic elaboration follow-up ("explain it in detail") after a
        # compound prior question ("What is fire insurance and what does it
        # cover?") is ambiguous about which part the user wants more on —
        # confirmed live: it silently picked one interpretation and answered
        # only that, with no way for the user to signal they wanted the other
        # part (or both). Detect this and ask the user which part they mean
        # via clickable chips, rather than guessing.
        # A single clarify chip click ("What does travel insurance cover?")
        # sends that exact question text as a normal message, indistinguishable
        # from the user typing it themselves — which meant it could semantic-
        # cache-hit the ORIGINAL broader compound question's cached answer
        # (they're highly similar) and silently replay it verbatim. Confirmed
        # live: clicking "What does travel insurance cover?" after "What is
        # travel insurance and what does it cover?" returned the exact same
        # cached brief answer instead of a fresh, focused one — defeating the
        # entire point of asking which part the user wanted. Detected the
        # same way the "both" reply is recovered: parse the two sub-questions
        # back out of our own clarify message and check for an exact match.
        _force_both_detailed = False
        _bypass_cache_for_chip_click = False
        if history:
            if _BOTH_REPLY_RE.match(question):
                _clarify_match = _CLARIFY_PARSE_RE.search(_history_last_assistant_turn(history))
                if _clarify_match:
                    _q1, _q2 = _clarify_match.group(1), _clarify_match.group(2)
                    question = f"{_q1.rstrip('?')}, and {_q2.rstrip('?')}?"
                    _force_both_detailed = True
            else:
                _clarify_match = _CLARIFY_PARSE_RE.search(_history_last_assistant_turn(history))
                if _clarify_match:
                    _q1, _q2 = _clarify_match.group(1), _clarify_match.group(2)
                    _norm_q = question.strip().rstrip("?").lower()
                    if _norm_q in (_q1.rstrip("?").lower(), _q2.rstrip("?").lower()):
                        _force_both_detailed = True
                        _bypass_cache_for_chip_click = True
            if _ELABORATION_FOLLOWUP_RE.search(question) and not _force_both_detailed:
                _last_q = _last_user_question(history)
                if _last_q:
                    _split = await _split_compound_question(_last_q)
                    if _split:
                        _q1, _q2 = _split
                        _clarify_text = _CLARIFY_TEMPLATE.format(q1=_q1.rstrip("?"), q2=_q2.rstrip("?"))
                        import json as _json_clarify
                        yield _clarify_text
                        yield "\n\n" + _json_clarify.dumps({
                            "sources": [],
                            "done": True,
                            "needs_human": False,
                            "clarify_options": [_q1, _q2, "Both, give me the full picture"],
                        })
                        return

        # ── Typo correction for retrieval query ───────────────────────────────
        # Apply _correct_typos() to build a corrected question used ONLY for
        # the retrieval_query, not for the final LLM prompt (original question
        # is preserved).  This ensures typos like "deductable" → "deductible"
        # before vector search.
        corrected_question = _correct_typos(question)

        # ── Follow-up reformulation ───────────────────────────────────────────
        # For short/vague follow-ups ("what about premiums?", "give me in detail",
        # "is it legal?") rewrite into a full standalone query using history.
        # Also catches longer pure acknowledgments ("oh okay that makes sense...")
        # that _is_likely_followup's word-count gate would otherwise treat as a
        # self-contained new question — see _is_conversational_reaction.
        _detected_as_followup = bool(
            history and (_is_likely_followup(question) or _is_conversational_reaction(question))
        )
        _is_followup = False
        if _detected_as_followup:
            _anchor_pattern = _get_dynamic_anchor_pattern(self.doc_pipeline.vector_store)
            _reformulated = await _reformulate_query(question, history, anchor_pattern=_anchor_pattern)
            if _reformulated is None:
                # Genuine failure (backend call errored/timed out, or returned
                # degenerate output) — distinct from the model correctly
                # returning the question unchanged (see _reformulate_query's
                # docstring). Only a real failure falls back to merging the
                # follow-up with the last assistant turn, via the same helper
                # ask() already uses for this. NOT just "the last user
                # question" (an earlier version of this fallback): that
                # DISCARDS the actual follow-up entirely — for a generic
                # modifier ("give me in detail") that's fine since it has no
                # content of its own, but for a substantive follow-up ("how
                # do I claim it?", "what does it not cover?") it silently
                # re-asks the PREVIOUS question and ignores what was actually
                # asked. Confirmed live: "What is motor insurance?" then "How
                # do I claim it?" — when reformulation failed, retrieval_query
                # became "What is motor insurance?" (Turn 1's exact text) via
                # the old fallback, which then KV-cache-hit Turn 1's cached
                # answer verbatim, so the claim question was never answered
                # at all. _reformulate_with_history keeps the actual question
                # (appended, not replaced) alongside topical context from the
                # last answer, so it degrades gracefully for both cases
                # instead of only the generic-modifier one.
                retrieval_query = question
                _merged = _reformulate_with_history(question, history)
                if _merged and _merged.lower() != question.lower():
                    retrieval_query = _merged
                    _is_followup = True
            else:
                retrieval_query = _reformulated
                if retrieval_query != question:
                    _is_followup = True
        else:
            retrieval_query = corrected_question

        # ── LLM-based query contextualization (replaces keyword follow-up detection) ──
        # Resolves pronouns and implicit references against recent conversation
        # history on every turn. The rewritten query is used for retrieval and
        # coverage checks; the original question is kept for the LLM prompt so
        # the model answers what the user actually asked.
        _contextualized = await _contextualize_query(retrieval_query, history)
        if _contextualized != retrieval_query:
            # Also log what the old keyword-based classifier would have said,
            # for side-by-side comparison in production logs before removal.
            _old_followup = _is_likely_followup(question) if history else False
            logger.info(
                "[CTX] %r → %r (old _is_likely_followup=%s)",
                retrieval_query, _contextualized, _old_followup,
            )
            retrieval_query = _contextualized

        # ── Keyword detailed check — must run BEFORE cache ───────────────────
        # So the cache key correctly separates brief vs. detailed for the same
        # topic ("what is health insurance" vs "explain health insurance in detail").
        # _force_both_detailed: a "both" reply to the compound-question
        # disambiguation above always wants the full picture on each part,
        # regardless of whether the combined question happens to match
        # _needs_detailed_answer()'s own signals.
        #
        # _resolve_modifier_intent also resolves has_simple/has_example here
        # (not just has_detail) — this must run before retrieval sizing below
        # anyway, so resolving all three together in one call (with one LLM
        # fallback round trip if needed, not up to three) lets the KV-cache
        # key and the instruction-injection block downstream reuse the same
        # result instead of independently re-running the fast-path-only
        # checks and missing whatever the LLM fallback just caught.
        _resolved_has_detail, _resolved_has_simple, _resolved_has_example = (
            await _resolve_modifier_intent(question)
        )
        _keyword_detailed = _resolved_has_detail or _force_both_detailed
        # Restored 8/6 -> 14/8 (2026-07-13, explicit user request after a
        # broad quality sweep). The 2026-07-10 reduction to 8/6 traded
        # quality for latency in a way that wasn't just "fewer results" —
        # turbovec_store.py's search() sets safe_k = min(2*top_k, count)
        # for the DENSE CANDIDATE POOL fed into reranking, so a smaller
        # top_k shrinks how many candidates the reranker even gets to see,
        # not just how many it returns. Confirmed live: for "Explain
        # liability insurance in detail", the single best-matching chunk
        # (insurance hb 1101.pdf p.20, cross-encoder rerank_score=0.79,
        # far above every other candidate) was completely absent from
        # top_k=8's results — it doesn't rank in the top-16 by raw dense/
        # BM25 similarity, so it never reached the reranker at all to get
        # the high score reflecting its real relevance. That's a
        # structural exclusion, not the reranker correctly judging it
        # weaker. Manifested as generic/padded-feeling detailed answers
        # for topics whose best chunk isn't a strong lexical/dense match
        # (e.g. liability, engineering insurance) even though the answer
        # format rules already explicitly permit shorter answers — the
        # model wasn't padding by choice, it was working with objectively
        # worse source material. Latency cost is real (this is what the
        # 2026-07-10 change traded away) but the user weighed answer
        # quality higher after seeing this evidence.
        # Widened 8/14 -> 12/18 (2026-07-17) — confirmed live this was still
        # a real retrieval-stage bottleneck, not just a reranking-quality
        # one: for "can I get medical insurance for my broken hand?" (brief
        # mode, top_k=8), the one chunk answering the actual question (a
        # pre-existing-injury exclusion clause) ranked #11 of 14 in the
        # RRF/dense+BM25 pre-rerank pool even in detailed mode's wider
        # setting — it never reached the reranker at all in brief mode,
        # so no amount of reranker-side improvement (see
        # _rerank_metadata_prefix in turbovec_store.py, the other half of
        # this fix) could have surfaced it. This stage uses cruder
        # embedding+keyword signals than the cross-encoder reranker that
        # runs after it, so it needs a wider net specifically to avoid
        # silently dropping a correct-but-differently-worded chunk before
        # the reranker ever gets a chance to judge it.
        _doc_top_k   = 18 if _keyword_detailed else 12
        # Trimmed 12/8 -> 8/5 (2026-07-13) — the final merged-and-reranked
        # pool actually sent to the LLM, kept deliberately separate from
        # _doc_top_k/_media_top_k above (which stay wide so the reranker
        # sees a big enough candidate pool to find the true best chunk).
        # Reranking still runs across the full wide pool before this final
        # cut, so this doesn't give up the structural-exclusion fix — it
        # only tightens which of the now-well-reranked chunks reach the
        # prompt. Verified live this reduces noise the wider candidate pool
        # can pull in: re-running "Explain liability insurance in detail"
        # (which had picked up an off-topic fire-insurance point right
        # after _doc_top_k was widened) with this tighter limit came back
        # clean, and "Explain engineering insurance in detail" correctly
        # gave 4 focused points instead of padding to 8 with vague filler.
        # Regression suite plus 8 more brief/detailed spot checks across
        # varied topics all held up or improved.
        _chunk_limit = 8 if _keyword_detailed else 5
        # Restored 4/3 -> 5/4 (2026-07-13, same request/evidence as the
        # _doc_top_k restore above) — this was trimmed by commit ec8bc3a
        # (2026-07-03) for the same "smaller candidate pool = faster
        # reranking, small chance of missing a good chunk" trade later
        # shown to actually bite in practice for _doc_top_k. video/webpage
        # search()'s internal reranking candidate pool is 2x this value
        # (safe_k = min(2*top_k, count)), so this alone controls how many
        # chunks each source reranks before picking the best ones.
        _media_top_k = 5 if _keyword_detailed else 4

        # Apply typo correction to the retrieval_query before KV cache and retrieval
        retrieval_query = _correct_typos(retrieval_query)

        # ── Metadata-based policy_type: hard pre-filter + soft re-rank ────────
        # classify_query_policy_type() uses a confidence bar suited to short
        # queries rather than classify_chunk_policy_type()'s >=2-hit
        # chunk-tuned bar (confirmed empirically: realistic queries only ever
        # score a single hit — the chunk-tuned bar never fires on a query at
        # all). When regex can't confidently name a type (most real queries
        # don't use the exact textbook phrase it looks for — "what's not
        # covered if my car is stolen" scores zero hits for every type),
        # _classify_query_policy_type_llm() gets one fast, cheap attempt
        # before falling back to "general" (no filtering).
        #
        # This WAS a hard search-time filter, then got removed 2026-07-16
        # after a live-confirmed false-negative: a genuinely relevant chunk
        # about home-insurance payouts was mistagged "health" (a single-
        # topic-per-chunk tag can't represent a chunk substantively
        # discussing several types — see project_live_upload_metadata_
        # pipeline_test), and the hard filter made it invisible to a "home
        # insurance" query no matter how relevant its content actually was.
        # Restored as a hard filter again (2026-07-17) now that: (a) the
        # LLM query-classification fallback above reduces how often
        # _query_policy_type itself is wrong to begin with, and (b) the
        # filter always includes "general" via $in below, so a chunk the
        # classifier genuinely couldn't confidently place — as opposed to
        # one confidently mistagged to the WRONG specific type — still
        # stays reachable. The residual risk this reopens (a confidently
        # mistagged, non-general chunk becoming invisible to the one query
        # that should have found it) is accepted as a known trade-off in
        # exchange for keeping wrong-type chunks out of the candidate pool
        # entirely, rather than just losing ties to on-topic chunks in it
        # (see the down-weight below, which still runs on top of this for
        # "general" vs. exact-type-matched chunks within the filtered pool).
        # Saved before the policy_type filter is merged in below, for the
        # standalone-retry path further down — that retry re-derives
        # retrieval from the RAW follow-up question text, which can be
        # about a genuinely DIFFERENT topic than whatever retrieval_query
        # (the follow-up-reformulated, prior-topic-anchored text) was
        # classified against. Applying a filter built from a stale,
        # wrong-topic classification there would block the exact rescue
        # that retry exists to attempt — it needs a clean, unfiltered
        # shot at the raw question, not the primary attempt's filter.
        _filter_meta_no_policy_type = filter_meta
        _query_policy_type = classify_query_policy_type(retrieval_query)
        if _query_policy_type == "general":
            _query_policy_type = await _classify_query_policy_type_llm(retrieval_query)
        if _query_policy_type != "general":
            _policy_type_filter = {"policy_type": {"$in": [_query_policy_type, "general"]}}
            filter_meta = {"$and": [filter_meta, _policy_type_filter]} if filter_meta else _policy_type_filter

        # ── KV cache lookup ───────────────────────────────────────────────────
        # Key includes reformulated query + intent flags so "why is X compulsory"
        # and "why is X compulsory, explain with example" never share a cache entry —
        # even though their retrieval_query is identical (example is a prompt modifier,
        # not a retrieval term, so it doesn't survive reformulation).
        _kv = self.doc_pipeline._cache
        _kv_sources = self.doc_pipeline._vector_store.list_sources()
        # Reuse the already-resolved intent (fast-path or LLM-fallback) from
        # above instead of independently re-running the fast-path-only
        # checks — otherwise a question the LLM fallback correctly caught as
        # "wants an example" would get cached under has_example=False here.
        _kv_has_example = _resolved_has_example
        _kv_has_simple  = _resolved_has_simple
        _kv_key = _kv.make_key(
            query=retrieval_query,
            top_k=_doc_top_k,
            use_hybrid=True,
            use_reranker=True,
            generate_answer=True,
            run_ragas=False,
            sources=_kv_sources,
            detailed=_keyword_detailed,
            has_example=_kv_has_example,
            has_simple=_kv_has_simple,
        )
        _kv_q_emb = None
        _kv_related_ctx = ""   # supplementary context from related cache entries
        # Testing-only escape hatch (default off, matches CONTAMINATION_TRACE's
        # env-gated pattern): QueryKVCache is a pure in-memory dict loaded once
        # at process start (kv_cache.py's self._data) — deleting the on-disk
        # JSON file between test requests does NOT clear it for the already-
        # running process, so repeat samples of the identical query text
        # (contamination_corpus_runner.py's --repeats) would otherwise all
        # replay the first generation instead of producing independent
        # samples of the model's real nondeterminism. Also skips the write
        # below so corpus-testing traffic never pollutes the real cache real
        # users hit.
        _disable_query_cache = os.getenv("DISABLE_QUERY_CACHE", "").strip().lower() in ("1", "true", "yes")
        _kv_hit = None if _disable_query_cache else _kv.get(_kv_key)
        if _kv_hit is None and not _disable_query_cache:
            try:
                _kv_q_emb = await asyncio.to_thread(
                    lambda: self.doc_pipeline._vector_store.embed_model.encode(
                        [retrieval_query], normalize_embeddings=True, show_progress_bar=False
                    )[0]
                )
                # Higher threshold for detailed queries so "life insurance in detail"
                # doesn't hit "health insurance in detail" just because topics are
                # close — was inverted (0.90, LOWER than the 0.92 base) until
                # 2026-07-13, doing the opposite of what this comment always said;
                # fixed to genuinely exceed the base threshold now that both moved
                # (base 0.92 -> 0.94, detailed 0.90 -> 0.97).
                _sem_thr = 0.97 if _keyword_detailed else None
                _sem_thr_actual = _sem_thr if _sem_thr is not None else 0.94

                # ── Semantic exact hit: same intent → serve directly ──────────
                _kv_hit = _kv.semantic_get(_kv_q_emb, threshold=_sem_thr)
                if _kv_hit is not None and _kv_hit.get("detailed") != _keyword_detailed:
                    _kv_hit = None
                if _kv_hit is not None and _kv_hit.get("has_example") != _kv_has_example:
                    _kv_hit = None
                if _kv_hit is not None and _kv_hit.get("has_simple") != _kv_has_simple:
                    _kv_hit = None

                # ── Semantic related: different question, overlapping topic ───
                # Collect entries in [0.80, sem_threshold) — related but not
                # identical.  Feed them to the LLM as supplementary context
                # alongside fresh KB chunks; never short-circuit the answer.
                # Lower bound raised 0.60 -> 0.80 (2026-07-13) — see
                # semantic_get_related's docstring in kv_cache.py for the live
                # measurement that motivated this; this was the actual
                # mechanism behind a stale answer's phrasing bleeding into
                # fresh generations for a genuinely different question.
                if _kv_hit is None:
                    _related = _kv.semantic_get_related(
                        _kv_q_emb,
                        lower_threshold=0.80,
                        upper_threshold=_sem_thr_actual,
                        top_k=2,
                    )
                    _rel_parts = []
                    _no_ans_phrases = (
                        "i don't have that", "don't have that specific",
                        "let me get one of our agents", "i can get one of our agents",
                        "i can get a human agent", "let me get a human agent",
                    )
                    _cur_type_text = f"{question} {retrieval_query}".lower()
                    for _rel in _related:
                        _q_txt = (_rel.get("query_text") or "").strip()
                        _a_txt = (_rel.get("answer") or "").strip()
                        # Skip entries whose intent flags differ from the current
                        # question — an example-based answer must not bleed into a
                        # simple-language request and vice versa.
                        if _rel.get("has_example") != _kv_has_example:
                            continue
                        if _rel.get("has_simple") != _kv_has_simple:
                            continue
                        # Skip entries naming a DIFFERENT specific insurance type
                        # than the current question. Structurally-similar
                        # sentences about different policy types ("get my money
                        # back if [X] insurance matures and nothing happened")
                        # score well inside the [0.80, threshold) related window
                        # on sentence shape alone — confirmed live: a motor
                        # insurance maturity question got answered with life
                        # insurance's "endowment assurance policy... on your
                        # death" language, copied near-verbatim from a cached
                        # life-insurance answer that scored 0.8-0.9 similar.
                        # query_text here is the reformulated/anchored text, so
                        # a follow-up correctly carries its established type.
                        _rel_type = _SPECIFIC_TYPE_RE.search(_q_txt.lower())
                        if _rel_type and _rel_type.group(1).lower() not in _cur_type_text:
                            continue
                        if _q_txt and _a_txt and not any(p in _a_txt.lower() for p in _no_ans_phrases):
                            _rel_parts.append(f"Q: {_q_txt}\nA: {_a_txt}")
                    if _rel_parts:
                        _kv_related_ctx = "\n\n".join(_rel_parts)
            except Exception:
                _kv_q_emb = None

        # A cache hit can replay an answer that was itself truncated mid-
        # generation when it was first cached (e.g. hit max_tokens). Serving
        # it verbatim to a differently-worded follow-up ("explain fire
        # insurance in detail" after "explain it in detail") silently
        # repeats the bad answer instead of getting a fresh, complete one —
        # confirmed live: a 2-sentence truncated answer with no closing line
        # got replayed unchanged for a reworded repeat of the same question.
        # DETAILED_GROUNDED_PROMPT always has the model end with a warm
        # closing line; its absence is a reliable signal generation was cut
        # short, so treat that as a cache miss and regenerate instead.
        if _kv_hit is not None and _keyword_detailed:
            _cached_ans_txt = (_kv_hit.get("answer") or "")
            _cached_looks_incomplete = not re.search(
                r"(hope that|let me know|feel free|hang tight|glad to|dig into any part|clears? it up)",
                _cached_ans_txt[-250:], re.IGNORECASE,
            )
            if _cached_looks_incomplete:
                logger.info(
                    "[ask_stream] discarding cache hit that looks truncated/incomplete (len=%d)",
                    len(_cached_ans_txt),
                )
                _kv_hit = None

        # Clicking a single clarify-disambiguation chip must always get a
        # fresh, focused answer — never a cache replay. A chip's question
        # text is semantically close enough to the original broader
        # compound question (that's precisely why disambiguation was
        # needed) to trigger a semantic cache hit on its cached answer,
        # confirmed live: clicking "What does travel insurance cover?"
        # right after "What is travel insurance and what does it cover?"
        # returned that exact same cached brief answer verbatim, silently
        # ignoring that the user specifically asked for more on just this
        # part.
        if _kv_hit is not None and _bypass_cache_for_chip_click:
            logger.info("[ask_stream] bypassing cache hit for clarify-chip click: %r", retrieval_query[:80])
            _kv_hit = None

        if _kv_hit is not None:
            import json as _json_s
            logger.info("[ask_stream] KV cache hit  query=%r detailed=%s", retrieval_query[:80], _keyword_detailed)
            _cached_answer = _kv_hit.get("answer", "")
            # A cache hit can replay an answer that was cached BEFORE the Rule 4
            # strip fix existed — the buggy two-part text would otherwise be
            # served verbatim forever until its TTL naturally expires. Check and
            # clean it here too, then persist the corrected version under the
            # current exact key so future exact-match hits are already clean.
            _cached_trust = _kv_hit.get("top_rerank", 0.0) >= 0.05
            _r4_cached = _strip_rule4_fallback(_cached_answer, trust_content=_cached_trust)
            if _r4_cached is not None:
                _cached_answer = _r4_cached
                logger.info("[ask_stream] stripped Rule4 fallback from cached answer (%d chars)", len(_r4_cached))
                try:
                    _kv.put(
                        _kv_key,
                        {**_kv_hit, "answer": _cached_answer},
                        query_embedding=_kv_q_emb,
                        query_text=retrieval_query,
                    )
                except Exception:
                    pass
            yield _cached_answer
            _cache_payload = {
                "sources": _kv_hit.get("sources", []),
                "done": True,
                "needs_human": False,
            }
            yield "\n\n" + _json_s.dumps(_cache_payload)
            return

        # Streaming path: parallel retrieval + LLM intent extraction.
        # _keyword_detailed is already computed above and is the only signal
        # used for detail level — an LLM-based detail classifier used to run
        # here too, but its result was never actually used (the LLM
        # over-classifies insurance questions as needing detail, making
        # every answer verbose), so it was purely wasted latency: a full
        # extra round-trip to the LLM server on every query for a result
        # nothing read. Removed.
        _t_preprocess_ms = round((time.time() - _t_request_start) * 1000)
        _t_retrieval_start = time.time()
        # Only fed to the actual search calls below, never to retrieval_query
        # itself — retrieval_query still drives the KV cache key, the lexical
        # coverage checks (fine either way, but unmodified matches user
        # expectations for what "the question" was), and prompt_question for
        # follow-ups further down. Keeping the expansion scoped to just the
        # search call avoids the appended text ever showing up in what the
        # model is told the user asked.
        _search_query = _expand_abbreviations(retrieval_query)
        if not document_filter:
            # Doc/video/webpage used to each pay their own separate
            # reranking call against the SAME shared process-wide
            # CrossEncoder — three fixed-overhead-dominated calls instead
            # of one, and each source always claimed its own fixed number
            # of final slots regardless of whether it had anything
            # genuinely relevant. Replaced with _retrieve_all_sources_
            # combined() (2026-07-13, evidence: isolated warm video
            # search+rerank measured ~7s on every call for a 28-chunk
            # store that contributed to ~1 of 30+ test answers that day)
            # — fetch raw candidates from every source, merge, and rerank
            # ONCE across the combined pool, so the final chunk_limit
            # slots go to whatever's actually most relevant regardless of
            # source, and a source with nothing relevant this time
            # contributes nothing on its own merits rather than by a
            # separate fixed quota.
            #
            # The LLM topic-extraction call has no CPU-reranking conflict
            # — it's a network call — so it still runs in genuine parallel
            # via its own task while retrieval proceeds.
            _topics_task = asyncio.create_task(_extract_intent_topics(question))
            all_chunks = await self._retrieve_all_sources_combined(
                _search_query, filter_meta, doc_top_k=_doc_top_k, summary_top_k=3,
                media_top_k=_media_top_k, chunk_limit=_chunk_limit,
            )
            llm_topics = await _topics_task
        else:
            doc_chunks, llm_topics = await asyncio.gather(
                self._retrieve_doc_chunks(_search_query, filter_meta, document_filter, doc_top_k=_doc_top_k, summary_top_k=3),
                _extract_intent_topics(question),
            )
            all_chunks = self._merge_chunks(doc_chunks)
        _t_retrieval_ms = round((time.time() - _t_retrieval_start) * 1000)
        detailed = _keyword_detailed

        # Prefer BGE rerank_score when available (set by rerank_documents).
        # Fall back to the raw retrieval similarity so non-reranked sources
        # (video, webpage) are still ordered reasonably.
        # Stage-1 boost chunks break ties upward: when two chunks share the
        # same rerank_score, the boost chunk (added because its document was
        # explicitly matched by the summary search) should win.
        #
        # Second layer on top of the hard pre-filter above: among the
        # document chunks that SURVIVED filtering (exact-type match or
        # "general" — a mismatched specific type never reaches this point
        # anymore), a "general"-tagged chunk still gets a soft scoring
        # penalty against an exact-type match. "general" is a deliberately
        # permissive tag covering genuinely cross-cutting reference
        # material, but that same permissiveness lets a chunk the
        # classifier just couldn't confidently place — not one that's
        # actually cross-cutting — win the reranker's top slot over
        # genuinely on-topic content purely on incidental lexical overlap.
        # Confirmed live: for "explain motor insurance in detail," a
        # "general"-tagged chunk about a government CROP-insurance scheme
        # for farmers scored 0.475 (rank #1 of 8) — higher than the actual
        # motor-insurance chunks (0.329, 0.318) — and that off-topic
        # content correlates with confirmed hallucinated detail in the
        # generated answer (see [[project_always_false_claim_corrections]]).
        # video/webpage chunks are exempt (source_type check below) since
        # they aren't covered by the hard pre-filter either.
        _TYPE_MISMATCH_DISCOUNT = 0.5

        # Mutated by the Mode-B fallback further down, read by
        # _effective_sort_score — a chunk that only made it into the pool
        # because the constrained search was already struggling shouldn't
        # then get penalized by the same mismatch logic that widening was
        # meant to route around. Content-hash keyed (matching _merge_chunks'
        # own dedup key) rather than id()-keyed, since rerank_documents()
        # isn't guaranteed to hand back the identical Document objects it
        # was given.
        _fallback_hashes: set = set()

        # Query-side open-vocabulary candidate, used only for the exact-match
        # exemption below. Gated on the pool actually containing a chunk with
        # a candidate label — the common case has nothing to compare against,
        # so this avoids paying an LLM call on every single query for a
        # comparison that usually can't matter.
        _query_candidate_type: Optional[str] = None
        if any(c.metadata.get("candidate_policy_type") for c in all_chunks):
            try:
                _query_candidate_type = await _classify_query_candidate_type_llm(retrieval_query)
            except Exception as exc:
                logger.debug("[CANDIDATE_TYPE] query-side gate failed: %s", exc)

        def _type_mismatched(c):
            _chunk_type = str(c.metadata.get("policy_type", "general")).lower()
            return (
                _query_policy_type != "general"
                and str(c.metadata.get("source_type", "document")).lower() == "document"
                and _chunk_type != _query_policy_type
            )

        def _candidate_mismatch(c):
            # A chunk with its OWN confidently-labeled open-vocabulary
            # candidate type (e.g. "pet_insurance" — not in the official
            # 12-type taxonomy at all) that disagrees with the query's own
            # candidate guess is much stronger negative evidence than the
            # generic "policy_type says general" case _type_mismatched
            # covers — we KNOW specifically what this chunk is about, and
            # we KNOW the query isn't about that. The soft 0.5x discount
            # below isn't enough to handle this: confirmed live, a pet-
            # insurance chunk (candidate_policy_type="pet_insurance") still
            # outranked the pool's only genuinely on-topic "home" chunk even
            # after a 0.5x discount, because that one real match itself
            # scored unusually low (0.0124) — general claims-process
            # boilerplate about pets scored 0.0572 raw purely on incidental
            # lexical overlap with "file a claim." Dropped outright instead
            # of discounted, for "How do I file a claim after water damage?"
            # leaking "just like visiting a vet for your pet" into a home-
            # insurance answer. Query's own candidate is checked (not just
            # "unset") so a real matching novel-type case is never dropped.
            return bool(
                str(c.metadata.get("source_type", "document")).lower() == "document"
                and c.metadata.get("candidate_policy_type")
                and c.metadata.get("candidate_policy_type") != _query_candidate_type
            )

        def _discount_exempt(c):
            # Two cases beyond "chunk isn't general at all" (that case never
            # reaches here — _type_mismatched is already False for it): (1)
            # both sides independently guessed the same open-vocabulary
            # topic, so the official "general" tag is known to be a
            # taxonomy gap, not genuine irrelevance; (2) this chunk only
            # exists in the pool because the fallback below had to widen
            # the net past the filter entirely.
            if hash(c.page_content[:200]) in _fallback_hashes:
                return True
            _chunk_candidate = c.metadata.get("candidate_policy_type")
            return bool(
                _chunk_candidate and _query_candidate_type and _chunk_candidate == _query_candidate_type
            )

        def _effective_sort_score(c):
            _base = c.metadata.get("rerank_score", c.metadata.get("similarity", 0))
            if _type_mismatched(c) and not _discount_exempt(c):
                return _base * _TYPE_MISMATCH_DISCOUNT
            return _base

        def _sort_and_truncate(chunks):
            # A mismatched/"general" chunk that scores below EVERY
            # genuinely type-specific match in the pool isn't losing a fair
            # ranking fight — it's just backfilling a slot chunk_limit
            # would otherwise leave unfilled. Confirmed live: a detailed-
            # mode query with only 7 real health-insurance chunks in its
            # retrieved pool still got an 8th, wrong-topic chunk (a newly-
            # uploaded pet insurance guide, tagged "general") forced into
            # the final context purely because chunk_limit=8 pads to a
            # fixed count — its discounted score (0.054) correctly ranked
            # below all 7 real matches (0.056-0.214), but "last place among
            # 8" still meant "included," and its claims-process wording
            # ended up echoed verbatim in a human health-insurance answer.
            # Drop these outright rather than let a fixed slot count
            # backfill with content that lost to every on-topic
            # alternative — a shorter, cleaner context beats a padded,
            # contaminated one.
            _type_specific_scores = [
                _effective_sort_score(c) for c in chunks
                if not _type_mismatched(c) or _discount_exempt(c)
            ]
            if _type_specific_scores:
                _min_type_specific = min(_type_specific_scores)
                chunks = [
                    c for c in chunks
                    if not (
                        _type_mismatched(c)
                        and not _discount_exempt(c)
                        and _effective_sort_score(c) < _min_type_specific
                    )
                ]
            chunks = sorted(
                chunks,
                key=lambda x: (
                    _effective_sort_score(x),
                    1 if x.metadata.get("stage1_boost") else 0,
                ),
                reverse=True,
            )
            return chunks[:_chunk_limit]

        _candidate_dropped = [c for c in all_chunks if _candidate_mismatch(c)]
        if _candidate_dropped:
            logger.info(
                "[ask_stream] candidate-type mismatch: dropping %s (query_candidate=%s)",
                [(c.metadata.get("source", ""), c.metadata.get("candidate_policy_type")) for c in _candidate_dropped],
                _query_candidate_type,
            )
            all_chunks = [c for c in all_chunks if c not in _candidate_dropped]

        all_chunks = _sort_and_truncate(all_chunks)

        # ── Hard reranker gate ────────────────────────────────────────────────
        # If the best reranker score across ALL retrieved chunks is below the
        # gate threshold, refuse immediately without calling the LLM at all.
        # Small 7B models ignore grounding instructions and answer from training
        # knowledge when the context is irrelevant, so gating here prevents that.
        # Scores are sigmoid-bounded [0,1] probabilities (see
        # _context_covers_query for empirical calibration notes) — but a
        # 0.2 threshold, tried previously, was measured to wrongly refuse
        # ~20% of legitimate KB questions: standard terms like "no claim
        # bonus" scored just 0.0024 and "free look period" scored 0.008 —
        # both far below 0.2 — while the actual bad case that motivated
        # raising this gate scored 0.062, HIGHER than those legitimate
        # answers. The score ranges for "genuinely relevant, oddly phrased"
        # and "borderline irrelevant" genuinely overlap; no single
        # threshold in between can separate them. This gate is kept only
        # as a backstop against results indistinguishable from random
        # noise (~0.00004-0.00006 in the same measurements) — the real
        # defense against ungrounded answers is the content-based checks
        # below (_context_covers_query, _enumeration_query_covered), which
        # check for actual word/entity presence rather than an ML
        # confidence score that isn't reliable at this granularity.
        import json as _json_s
        _rerank_gate = float(os.getenv("RERANK_GATE_THRESHOLD", "0.0005"))
        _top_rerank = max(
            (c.metadata.get("rerank_score", float("-inf"))
             for c in all_chunks if hasattr(c, "metadata") and "rerank_score" in c.metadata),
            default=float("-inf"),
        )

        # ── Mode-B retrieval fallback ────────────────────────────────────────
        # Closes the gap the discount above can't reach: when ingestion and
        # query-time classification each confidently but wrongly force-fit
        # the same novel topic into two DIFFERENT existing types, the hard
        # policy_type filter (built above, "$in": [_query_policy_type,
        # "general"]) excludes the correct chunks before they're ever
        # scored — there's no "general" tag or candidate label to catch,
        # because neither classifier ever produced an out-of-vocabulary
        # answer for this specific case. The only observable signal is the
        # symptom (a weak top score against the filtered pool), not the
        # cause, so this doesn't try to detect a mismatch — it just widens
        # the net when the constrained search already looks like it's
        # struggling, and keeps the result only if it's actually better.
        # Scoped the same way the two existing retry tiers below are
        # (not document_filter): a query with an explicit document scope
        # has already told retrieval exactly where to look.
        _fallback_threshold = float(os.getenv("RETRIEVAL_FALLBACK_THRESHOLD", "0.01"))
        if (
            not document_filter
            and _query_policy_type != "general"
            and _top_rerank < _fallback_threshold
        ):
            try:
                _unfiltered_raw = await asyncio.to_thread(
                    self.doc_pipeline._vector_store.search,
                    _search_query, top_k=_doc_top_k, use_hybrid=True, use_reranker=False,
                    filter_metadata=_filter_meta_no_policy_type,
                )
                if _unfiltered_raw:
                    _existing_hashes = {hash(c.page_content[:200]) for c in all_chunks}
                    _merged_pool = self._merge_chunks(all_chunks + _unfiltered_raw)
                    _new_hashes = {hash(c.page_content[:200]) for c in _merged_pool} - _existing_hashes
                    # Rerank the merged pool ONCE rather than fetching and
                    # reranking the unfiltered results separately — pool size
                    # dominates rerank latency far more than whether a filter
                    # was applied, so this is "rerank a bigger pool once,"
                    # not "rerank twice."
                    _merged_pool = await asyncio.to_thread(
                        self.doc_pipeline._vector_store.rerank_documents,
                        _search_query, _merged_pool, _chunk_limit,
                    )
                    _fallback_hashes.update(_new_hashes)
                    _merged_pool = _sort_and_truncate(_merged_pool)
                    _merged_top = max(
                        (c.metadata.get("rerank_score", float("-inf"))
                         for c in _merged_pool if hasattr(c, "metadata") and "rerank_score" in c.metadata),
                        default=float("-inf"),
                    )
                    if _merged_top > _top_rerank:
                        logger.info(
                            "[ask_stream] Mode-B fallback: filtered top=%.4f -> merged top=%.4f, adopting wider pool",
                            _top_rerank, _merged_top,
                        )
                        all_chunks = _merged_pool
                        _top_rerank = _merged_top
                    else:
                        _fallback_hashes.difference_update(_new_hashes)
            except Exception as exc:
                logger.debug("[ask_stream] Mode-B fallback search failed: %s", exc)

        if not document_filter and _top_rerank < _rerank_gate and not _pasted_grounds_answer:
            logger.info(
                "[ask_stream] Reranker gate: top=%.3f < gate=%.3f — not in KB",
                _top_rerank, _rerank_gate,
            )
            # This refusal still does real retrieval+reranking work before
            # concluding nothing is relevant — unlike the trivial early
            # returns (greeting, exact-cache-hit) the module comment above
            # _t_request_start deliberately skips instrumenting, this one
            # is genuinely worth measuring (it's what a latency baseline's
            # "refusal" case actually exercises). Reuses the same TIMING
            # line format so a log-scraping harness needs only one regex.
            # Nothing streams on this path, so TTFT is the whole request —
            # one timestamp for both, not two near-identical time.time()
            # calls.
            _t_total_ms = round((time.time() - _t_request_start) * 1000)
            _t_ttft_ms = _t_total_ms
            # `other` MUST be computed the same way the main TIMING line
            # does at the end of this function, not hardcoded to 0 — it is
            # defined there as "total minus the named phases", i.e. exactly
            # the untimed glue and the fallback/standalone-retry tiers.
            # Passing a literal 0 asserts there is no unaccounted time,
            # which on a refusal is badly wrong: measured live, ~48% of a
            # refusal's wall clock sits in the pre-refusal retry cascade
            # and reporting 0 hid all of it.
            _t_other_ms = _t_total_ms - sum(
                v for v in (_t_retrieval_ms, _t_grounding_ms) if v is not None
            )
            logger.info(
                "[ask_stream] TIMING total=%dms ttft=%s retrieval=%s grounding=%s llm=%s other=%dms "
                "preprocess=%s promptbuild=%s postllm=%s detailed=%s query=%r question=%r",
                _t_total_ms,
                f"{_t_ttft_ms}ms",
                f"{_t_retrieval_ms}ms" if _t_retrieval_ms is not None else "n/a",
                "n/a", "n/a", _t_other_ms,
                f"{_t_preprocess_ms}ms" if _t_preprocess_ms is not None else "n/a",
                "n/a", "n/a",
                _keyword_detailed,
                retrieval_query[:80],
                question[:80],
            )
            yield (
                "Hmm, I don't have that specific information in my knowledge base right now. "
                "Let me get one of our agents on it, they'll be able to help you better! 😊"
            )
            yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": True})
            return

        # A single strong chunk clearing the gate above must not let unrelated
        # weak chunks ride along into context — every chunk the LLM sees has
        # to individually clear the relevance bar, not just "someone in the
        # retrieved pool did". Only drop chunks that HAVE a rerank_score below
        # gate; anything without one (shouldn't happen now that video/webpage
        # are reranked too) is left alone rather than silently dropped.
        if not document_filter:
            all_chunks = [
                c for c in all_chunks
                if c.metadata.get("rerank_score", _rerank_gate) >= _rerank_gate
            ]

        # Reorder so chunks naming the query's specific insurance type sort
        # first — matters here specifically because _verify_grounding below
        # only looks at the first _GROUNDING_CONTEXT_CHARS (3000) of the
        # joined context, in whatever order all_chunks is already in. If a
        # wrong-topic-but-similarly-worded chunk from a DIFFERENT insurance
        # type outranks the real one, the correct content can get pushed
        # past that 3000-char cutoff and never even reach the grounding
        # check. Confirmed live: "are all types of illness covered under
        # health insurance?" top-scored (0.78) a TRAVEL insurance exclusion
        # list — correctly rejected by _verify_grounding as not actually
        # about health insurance — while a genuine health-insurance
        # exclusion chunk (cosmetic/aesthetic treatment, 0.10) sat lower in
        # the same pool and never got evaluated. Reordering doesn't change
        # what's included, only gives the correct-topic content a chance to
        # be seen by the checks that decide whether to answer at all.
        all_chunks = _prioritize_topic_chunks(retrieval_query, all_chunks)

        # Run coverage check on the pre-compression chunks so that video/webpage
        # chunks filling the budget first don't cause doc-chunk terms to go missing.
        # Lexical checks run alongside the semantic _verify_grounding() backstop
        # via asyncio.gather() rather than after it, so the LLM round-trip isn't
        # serialized behind work that's already fast. Both must pass.
        #
        # full_context (the labeled, budget-compressed version used for the
        # actual generation prompt) isn't built until later in this function —
        # unlike ask(), where it already exists by this point. Build a plain,
        # uncompressed join of all_chunks here instead, just for the grounding
        # check; it doesn't touch full_context or the ContextCompressor at all.
        topics_for_coverage = None if _detected_as_followup else (llm_topics or None)
        async def _lexical_covered():
            return (
                _context_covers_query(retrieval_query, all_chunks, llm_topics=topics_for_coverage)
                and _quoted_comparison_covered(retrieval_query, all_chunks)
                and _enumeration_query_covered(retrieval_query, all_chunks)
            )
        _t_grounding_start = time.time()
        _lex_ok, _semantically_grounded = await asyncio.gather(
            _lexical_covered(),
            _verify_grounding_any_chunk(retrieval_query, all_chunks),
        )
        _t_grounding_ms = round((time.time() - _t_grounding_start) * 1000)
        _t_promptbuild_start = time.time()
        # High-confidence bypass: _verify_grounding_any_chunk is a small-
        # model YES/NO judgment call, built specifically because this
        # project's generation model unreliably hallucinates from training
        # knowledge when just handed weak/irrelevant context (see the
        # reranker-gate comment above) — but the SAME judgment call is
        # itself occasionally wrong in the opposite direction. Confirmed
        # live: "What is the maximum compensation for legal expenses under
        # travel insurance?" retrieved the exact answer (rerank_score=0.976,
        # lex_ok=True) yet _verify_grounding_any_chunk said NO on both its
        # full-context and top-chunk-alone paths, and the SAME clean
        # passage handed to the identical judgment call in isolation
        # (no retrieval noise) got a correct YES 4/4 times — the retrieved
        # content plainly was there, the gate was simply wrong that once.
        # Only bypass at 0.9+ ("near-certain" per this file's own score
        # calibration notes above), well clear of the 0.78 wrong-topic case
        # already documented a few lines up ("...top-scored (0.78) a TRAVEL
        # insurance exclusion list...correctly rejected") — that
        # documented false-positive risk zone sits comfortably below this
        # threshold, so this bypass doesn't reopen it. Still requires
        # _lex_ok too — a very high rerank score alone isn't trusted on its
        # own, matching the existing single_topic bypass inside
        # _context_covers_query, which also never trusts the score in
        # isolation.
        _grounding_bypass = _lex_ok and _top_rerank >= 0.9
        # Symmetric bypass for the OPPOSITE failure: _lex_ok itself is the
        # weakest of the three signals here (whole-word matching against a
        # hand-tuned 5-char-prefix heuristic — see _word_matches), and it can
        # fail even when the other two, stronger signals agree. Confirmed
        # live: "Does home protection cover jewellery and electronics inside
        # the house?" retrieved the exact right chunk (top_rerank=0.9895) and
        # _verify_grounding_any_chunk — the LLM having actually read the
        # content — correctly said YES, yet _lex_ok was False because the
        # extracted topic phrase "home protection" got split into the words
        # "home" + "protection", and "home" (a short, <5-char topic word, so
        # only an EXACT token match counts — see _word_matches) never occurs
        # as a standalone token in that chunk, only inside "Homeowners". Two
        # independent signals — an embedding-based reranker and an LLM that
        # read the actual text — agreeing at a near-certain bar is strong
        # enough evidence on its own; a tokenization artifact in the third,
        # admittedly-weaker proxy signal shouldn't be able to override both.
        _semantic_high_confidence_bypass = _semantically_grounded and _top_rerank >= 0.9
        ctx_covered = (
            _grounding_bypass
            or _semantic_high_confidence_bypass
            or (_lex_ok and _semantically_grounded)
        )
        if _grounding_bypass and not _semantically_grounded:
            logger.info(
                "[ask_stream] grounding bypass: top_rerank=%.3f >= 0.9, lex_ok=True — "
                "overriding a NO from _verify_grounding_any_chunk",
                _top_rerank,
            )
        if _semantic_high_confidence_bypass and not _lex_ok:
            logger.info(
                "[ask_stream] semantic bypass: top_rerank=%.3f >= 0.9, "
                "semantically_grounded=True — overriding a False from _lex_ok",
                _top_rerank,
            )

        _dropped_terms_note = None
        _answered_via_standalone_retry = False
        if not ctx_covered and not document_filter and not _pasted_grounds_answer:
            # Before refusing, retry once with an LLM-cleaned, retrieval-
            # optimized rewrite — fixes general spelling mistakes and strips
            # specific qualifiers (a country/city name, "affordable"/"cheap"/
            # "best") that can tank the reranker's score even though the KB
            # covers the core topic well (see _vllm_clean_query's docstring
            # for measured examples). If the cleaned query now passes, the
            # answer proceeds using it — but ONLY when what got dropped was
            # an emphasis word (affordable/cheap/best), not a proper noun
            # (country/city/provider name). Confirmed live: a dropped
            # proper noun means the KB has NO factual content about that
            # specific thing at all — "which insurers cover travel to South
            # Africa" cleaned down to "travel insurance", which the KB
            # covers well, and the prompt-injected instruction to
            # "explicitly say up front" that South Africa specifically
            # isn't covered was silently ignored by the model 100% of the
            # time it was actually exercised (this codebase's small vLLM
            # model has repeatedly proven unreliable at honoring buried
            # prompt instructions — see the nominee-caveat and Groq-
            # hallucination investigations earlier this session). Rather
            # than trust it to remember, a dropped proper noun is treated
            # the same as "not covered" — the user gets a clean refusal/
            # handoff instead of a generic answer standing in for the
            # specific thing they actually asked about.
            _cleaned_query, _dropped, _dropped_proper_noun = await _vllm_clean_query(retrieval_query)
            if (
                _cleaned_query
                and not _dropped_proper_noun
                and _cleaned_query.strip().lower() != retrieval_query.strip().lower()
            ):
                _fallback_chunks = await self._retrieve_doc_chunks(
                    _cleaned_query, filter_meta, document_filter, doc_top_k=_doc_top_k, summary_top_k=2,
                )
                _fallback_top = max(
                    (c.metadata.get("rerank_score", float("-inf"))
                     for c in _fallback_chunks if hasattr(c, "metadata") and "rerank_score" in c.metadata),
                    default=float("-inf"),
                )
                if _fallback_top >= _rerank_gate:
                    _fallback_lex_ok = (
                        _context_covers_query(_cleaned_query, _fallback_chunks, llm_topics=None)
                        and _quoted_comparison_covered(_cleaned_query, _fallback_chunks)
                        and _enumeration_query_covered(_cleaned_query, _fallback_chunks)
                    )
                    _fallback_sem_grounded = await _verify_grounding_any_chunk(
                        _cleaned_query, _fallback_chunks, backend_override="vllm",
                    )
                    if _fallback_lex_ok and _fallback_sem_grounded:
                        # Same gap as the standalone-retry tier below:
                        # _retrieve_doc_chunks reranks but never applies
                        # _effective_sort_score, so a "general"-tagged,
                        # weaker-than-every-real-match chunk that squeaked
                        # past filter_meta here would skip the discount/
                        # exclusion pass entirely if assigned directly.
                        all_chunks = _sort_and_truncate(_fallback_chunks)
                        retrieval_query = _cleaned_query
                        ctx_covered = True
                        _dropped_terms_note = _dropped

        # ── Standalone-retry tier for misclassified follow-ups ──────────────
        # When the follow-up classification/reformulation was wrong (the
        # question looked like a pronoun-dependent follow-up but was actually
        # a fresh standalone query), retry the ORIGINAL question text against
        # the KB directly — ignoring history and contextualization entirely —
        # before refusing.  This is a CRAG-style fallback: a lightweight
        # classifier (heuristic) may be wrong; we verify by trying the
        # alternate strategy and checking if the result actually grounds.
        #
        # Only valid when the question COULD plausibly be self-contained.
        # If it contains an explicit unresolved pronoun (_FOLLOWUP_SIGNALS —
        # "it", "that", "this", "them"...), the "misclassified" premise
        # can't hold: a genuinely standalone question wouldn't have a
        # dangling pronoun with no antecedent in the first place. Retrying
        # such text standalone strips the one content word that mattered
        # ("it" contributes nothing after stopword-filtering) and leaves
        # something like bare "claim" — which trivially lexically+
        # semantically matches whatever generic claims content ranks
        # highest in the KB. Confirmed live: "How to claim it?" after a
        # term-insurance question retried as bare "claim", passed both the
        # lexical AND semantic grounding checks 8/8 times in isolation
        # testing (not a flaky/occasional pass — reliably wrong every
        # time), and answered with confidently-blended motor-insurance
        # claims content (driving license, FIR for vehicle theft) that has
        # nothing to do with term insurance — a wrong-topic answer, strictly
        # worse than the refusal it was replacing.
        # A "point N" reference (_extract_point_number returning non-None) is
        # just as structurally follow-up-dependent as a pronoun, even with no
        # pronoun word present — "explain point 2" only means something
        # relative to whatever numbered list is in history. Confirmed live:
        # "can you explain 2 point in simple language with example" has no
        # pronoun, so it passed the guard below unchanged, retried standalone
        # on the bare topic-less text, and confidently answered about
        # subrogation — a completely unrelated topic pulled from a generic
        # "general insurance principles" section that ranks well for almost
        # any vague "explain simply with an example" phrasing. Same failure
        # mode as the pronoun case, just without a pronoun to catch it.
        #
        # A THIRD variant of the same underlying gap, found via live browser
        # testing (2026-07-15): a bare generic-modifier follow-up with no
        # pronoun AND no point reference — "can you give me an example" (no
        # "it"/"that", no "point N") passed both guards above unchanged.
        # Debug-traced live: _reformulate_query correctly anchored it to
        # "Can you provide a real-life example of motor insurance?", but that
        # correctly-anchored query's ctx_covered check failed (an "example"-
        # style question's retrieved content doesn't always satisfy the
        # lexical/semantic coverage bar as cleanly as a factual question
        # does), so this retry fired anyway — retrying the BARE "can you give
        # me an example" against the KB directly, which confidently matches
        # almost ANY illustrative-example chunk (fire insurance, subrogation,
        # whatever ranks highest) regardless of the actual prior topic, and
        # that wrong-topic match passed its own grounding checks. Same root
        # cause as the pronoun/point-ref cases: this retry is only safe when
        # the raw question carries enough substantive content of its own to
        # plausibly BE a genuine standalone question — a bare "explain more"/
        # "give an example"/"in simple terms" modifier with no insurance
        # vocabulary of its own carries none, exactly like a dangling pronoun
        # or a "point N" reference. Reuses the identical modifier+no-
        # vocabulary test _is_likely_followup() already uses for the same
        # judgment, rather than inventing a new one.
        _question_tokens = {w.lower().strip('?.,!') for w in question.split()}
        _has_unresolved_pronoun = bool(_question_tokens & _FOLLOWUP_SIGNALS)
        _has_unresolved_point_ref = _extract_point_number(question) is not None
        _q_lower_for_modifier_check = question.lower().strip()
        _is_bare_modifier_only = (
            any(sig in _q_lower_for_modifier_check for sig in (_DETAIL_SIGNALS | _SIMPLE_SIGNALS))
            or _wants_example(_q_lower_for_modifier_check)
            or _DETAIL_PATTERN.search(_q_lower_for_modifier_check)
        ) and not re.search(
            r"\b(insurance|policy|premium|deductible|coverage|claim|health|medical|"
            r"life|motor|travel|home|vehicle|accident|liability|rider|annuity|pension)\b",
            _q_lower_for_modifier_check,
        )
        if (
            not ctx_covered and not document_filter and not _pasted_grounds_answer
            and _detected_as_followup and not _has_unresolved_pronoun and not _has_unresolved_point_ref
            and not _is_bare_modifier_only
        ):
            _standalone_chunks = await self._retrieve_doc_chunks(
                question, _filter_meta_no_policy_type, document_filter, doc_top_k=_doc_top_k, summary_top_k=2,
            )
            _standalone_top = max(
                (c.metadata.get("rerank_score", float("-inf"))
                 for c in _standalone_chunks if hasattr(c, "metadata") and "rerank_score" in c.metadata),
                default=float("-inf"),
            )
            if _standalone_top >= _rerank_gate:
                # Check standalone against the ORIGINAL question text, not
                # the followup-reformulated retrieval_query.
                _standalone_lex_ok = (
                    _context_covers_query(question, _standalone_chunks, llm_topics=None)
                    and _quoted_comparison_covered(question, _standalone_chunks)
                    and _enumeration_query_covered(question, _standalone_chunks)
                )
                _standalone_sem_grounded = await _verify_grounding_any_chunk(question, _standalone_chunks)
                if _standalone_lex_ok and _standalone_sem_grounded:
                    # This retry deliberately searches unfiltered by
                    # policy_type (see _filter_meta_no_policy_type above) —
                    # the whole point is escaping a possibly-wrong prior-
                    # topic-anchored classification, so a hard filter here
                    # would defeat the retry itself. But that leaves the
                    # same soft-discount/exclusion pass every other path
                    # gets as the only thing standing between this pool and
                    # the exact "general"-tagged, wrong-topic chunk problem
                    # fixed above — _retrieve_doc_chunks does its own
                    # reranking but never applies _effective_sort_score, so
                    # a chunk this unfiltered search pulls in gets no
                    # discount and no exclusion check at all otherwise.
                    # Reclassify against `question` itself (not the stale
                    # follow-up-reformulated retrieval_query) since that's
                    # exactly what this retry just proved is the right
                    # anchor going forward — matches the retrieval_query
                    # reassignment right below.
                    _query_policy_type = classify_query_policy_type(question)
                    if _query_policy_type == "general":
                        _query_policy_type = await _classify_query_policy_type_llm(question)
                    all_chunks = _sort_and_truncate(_standalone_chunks)
                    retrieval_query = question
                    ctx_covered = True
                    _detected_as_followup = False  # answer as a fresh question, not a followup
                    _answered_via_standalone_retry = True

        if not ctx_covered and not document_filter and not _pasted_grounds_answer:
            # Before refusing: "what does X mean?" / "define X" asked right
            # after X appeared in the bot's own last answer is answerable
            # even when X has no dedicated KB entry (e.g. "discount" in "a
            # no-claim bonus is a discount on your premium") — X is an
            # ordinary word, not an insurance concept, so retrieval finds
            # nothing and the checks above correctly say "not covered by the
            # KB". But refusing here is still wrong: the model can define an
            # ordinary word from its own training knowledge just fine, it
            # only needs to know WHAT WAS JUST SAID so the definition lands
            # in the right context (premium discount, not a retail discount)
            # instead of a generic, disconnected dictionary answer. Gated on
            # the word actually appearing in the immediately preceding
            # assistant turn — otherwise this would just be an ungrounded
            # general-knowledge answer with no relation to what was asked.
            _meaning_word = _extract_meaning_query_word(question)
            if (
                _meaning_word
                and _last_assistant_turn
                and re.search(r"\b" + re.escape(_meaning_word.lower()) + r"\b", _last_assistant_turn.lower())
            ):
                _meaning_prompt = (
                    f"You just told the user this:\n\"{_last_assistant_turn}\"\n\n"
                    f"They're now asking what \"{_meaning_word}\" means, as you just used it above. "
                    f"Explain it in 1-2 short sentences, plain conversational language, "
                    f"specifically in that same context — not a generic, unrelated dictionary "
                    f"definition. You can use your own general knowledge; this word doesn't need "
                    f"to come from any document."
                )
                _meaning_answer = await _backend_completion(_meaning_prompt, max_tokens=120, timeout=10.0)
                if _meaning_answer and _meaning_answer.strip():
                    yield _meaning_answer.strip()
                    yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": False})
                    return
            # This is the grounding-check refusal — retrieval found something
            # that cleared the reranker gate, but lexical/semantic coverage
            # (and every retry tier above) still said no. Distinct exit point
            # from the reranker-gate refusal above (that one never even
            # reaches grounding); confirmed live this is the path the
            # latency plan's own fixed refusal test case ("1998 Yugo GV in
            # Alaska") actually hits, not the reranker-gate one — so it
            # needs its own TIMING line for the same reason that one did.
            # Reuses the identical format string; retrieval+grounding are
            # both real here (unlike the reranker-gate case), llm/promptbuild
            # never ran.
            #
            # `other` is where this path's single largest cost lives, so it
            # is computed exactly as the main TIMING line does rather than
            # hardcoded — see the same note at the reranker-gate refusal.
            # _t_promptbuild_start is set right after the grounding gather
            # but _t_promptbuild_ms is only assigned much later (past this
            # return), so everything between — _vllm_clean_query, the
            # fallback retrieval+rerank, the standalone-retry tier, and
            # their two extra _verify_grounding_any_chunk round-trips — is
            # untimed and lands here. Measured live: ~5.8s of an ~11.9s
            # refusal, i.e. more than retrieval, previously logged as 0.
            _t_total_ms = round((time.time() - _t_request_start) * 1000)
            _t_ttft_ms = _t_total_ms
            _t_other_ms = _t_total_ms - sum(
                v for v in (_t_retrieval_ms, _t_grounding_ms) if v is not None
            )
            logger.info(
                "[ask_stream] TIMING total=%dms ttft=%s retrieval=%s grounding=%s llm=%s other=%dms "
                "preprocess=%s promptbuild=%s postllm=%s detailed=%s query=%r question=%r",
                _t_total_ms,
                f"{_t_ttft_ms}ms",
                f"{_t_retrieval_ms}ms" if _t_retrieval_ms is not None else "n/a",
                f"{_t_grounding_ms}ms" if _t_grounding_ms is not None else "n/a",
                "n/a", _t_other_ms,
                f"{_t_preprocess_ms}ms" if _t_preprocess_ms is not None else "n/a",
                "n/a", "n/a",
                _keyword_detailed,
                retrieval_query[:80],
                question[:80],
            )
            yield (
                "Hmm, I don't have that specific information in my knowledge base right now. "
                "Let me get one of our agents on it, they'll be able to help you better! 😊"
            )
            yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": True})
            return

        # Dynamic context budget — scale back when history is long so the total
        # prompt (template + history + context + answer) stays within the model's
        # context window (~4096 tokens for Qwen2.5-7B; use 3900 as safe ceiling).
        # 4 chars ≈ 1 token (Qwen SentencePiece approximation).
        _MAX_INPUT_TOKENS = 3900
        # Picks the estimate for whichever prompt template this request will
        # actually use (same document_filter/detailed branching as the
        # prompt-selection block further down) — was a single flat 700-token
        # guess for all three, which drifted badly out of date and caused a
        # real production crash; see the constants' definition near the
        # prompt_template import for the full story.
        if document_filter:
            _PROMPT_TEMPLATE_TOKENS = _STRICT_PROMPT_TOKENS_EST
        elif detailed:
            _PROMPT_TEMPLATE_TOKENS = _DETAILED_PROMPT_TOKENS_EST
        else:
            _PROMPT_TEMPLATE_TOKENS = _CONVERSATIONAL_PROMPT_TOKENS_EST
        _output_reserve = 1500 if detailed else 300
        _history_tokens_est = len(history) // _CHARS_PER_TOKEN if history else 0

        # If template + history + output_reserve alone already leave no
        # room even for the 300-token context floor below, the floor stops
        # context from shrinking further but does NOT stop the total from
        # exceeding the model's window — confirmed live: this is exactly
        # what crashed a real, long-running user conversation (session
        # continued well past a few turns) with an "internal error" on a
        # point-reference follow-up. Truncate history from the OLDEST
        # turns (keep the most recent ones — history is used for context on
        # follow-ups, and the most recent turns matter most for that) until
        # there's guaranteed room, rather than trusting the floor alone.
        _min_reserve_without_history = _PROMPT_TEMPLATE_TOKENS + 300 + _output_reserve
        _max_history_tokens = max(0, _MAX_INPUT_TOKENS - _min_reserve_without_history)
        _max_history_chars = _max_history_tokens * _CHARS_PER_TOKEN
        if history and len(history) > _max_history_chars:
            history = history[-_max_history_chars:]
            _first_newline = history.find("\n")
            if _first_newline != -1:
                history = history[_first_newline + 1:]
            _history_tokens_est = len(history) // _CHARS_PER_TOKEN

        _context_token_budget = max(
            300,  # always keep at least a bit of context
            _MAX_INPUT_TOKENS - _PROMPT_TEMPLATE_TOKENS - _history_tokens_est - _output_reserve,
        )
        _context_budget = min(6000, _context_token_budget * _CHARS_PER_TOKEN)  # tokens → chars

        # Drop near-zero-relevance VIDEO/WEBPAGE chunks before fair-share
        # compression gives every SURVIVING chunk a guaranteed slice of the
        # context budget. Confirmed live: a "how to claim life insurance"
        # query's candidate pool included YouTube transcripts about health
        # and car insurance claims scoring 0.004-0.015 on the reranker,
        # versus 0.15-0.91 for the genuinely relevant life-insurance
        # chunks — a decisive 10x+ gap. Before this filter, fair-share
        # (previous commit) guaranteed those near-zero-relevance chunks a
        # slice too, and their off-topic claim-procedure language bled
        # into the generated answer ("whether it's for life insurance or
        # health insurance"). The old greedy-fill compression had
        # accidentally masked this by crushing low-ranked chunks to
        # nothing — fair-share fixed that for genuinely relevant content
        # but also removed that accidental protection for actual noise.
        #
        # Scoped to video/webpage sources ONLY, never document/PDF chunks
        # — confirmed live this matters: a 0.05 threshold applied to ALL
        # source types wrongly excluded the correct "relatives" answer (a
        # PDF chunk scoring 0.028, a previously-fixed, documented case of
        # a genuinely-relevant chunk the reranker underscores because its
        # key sentence is buried past an unrelated opening section) while
        # every confirmed noise chunk in testing was a YouTube transcript.
        # Unlike the curated, structured PDF course material, video
        # transcripts run continuously across unrelated topics, so a
        # low score on them is a much more reliable "actually irrelevant"
        # signal than the same score on a document chunk.
        #
        # Raised 0.05 -> 0.15 (2026-07-16) — 0.05 was too permissive to
        # actually screen out noise: confirmed live, a "How to file Car
        # Insurance Claim UAE" video scored 0.069 against "what should i do
        # to file a claim if my machinery breaks down during construction"
        # (an ENGINEERING insurance question, no relation to motor claims)
        # — "file a claim"/"breaks down" surface-level phrasing overlap was
        # enough to clear 0.05 despite the video being about a completely
        # different insurance type. The model faithfully wove the video's
        # actual content (WhatsApp support, Emirates Police app, driving
        # license) into the answer as if it applied to machinery breakdown.
        # 0.15 clears that false match with real margin while staying well
        # below genuinely relevant video scores (0.3+ in other confirmed
        # cases), so on-topic video content still surfaces normally.
        _MIN_RERANK_SCORE = 0.15
        _NON_DOCUMENT_SOURCE_TYPES = {"video", "youtube_transcript", "youtube", "webpage", "web"}

        def _keep_chunk(c) -> bool:
            if str(c.metadata.get("source_type", "document")).lower() not in _NON_DOCUMENT_SOURCE_TYPES:
                return True
            _score = c.metadata.get("rerank_score")
            return _score is None or _score >= _MIN_RERANK_SCORE

        _relevant_chunks = [c for c in all_chunks if _keep_chunk(c)]
        if _relevant_chunks:
            all_chunks = _relevant_chunks

        # Full, uncompressed chunk text — kept alongside the (possibly
        # compressed) prompt context below specifically for the post-
        # generation grounding checks further down (_point_grounded,
        # currency/qualifier-mismatch, refund-scope-mismatch, denial-claim
        # checks). Those checks measure whether the model's paraphrase
        # shares enough vocabulary with what was actually retrieved — but
        # compress_to_budget can shrink a multi-hundred-word chunk down to
        # a single ~100-char sentence when many chunks compete for a
        # shared budget (confirmed live: 8 detailed-mode chunks compressed
        # to 64-114 chars each, ~1300 chars total). The LLM was still shown
        # only the compressed version and is expected to answer from it —
        # but a genuinely correct paraphrase can reasonably restate detail
        # from elsewhere in the SAME source chunk that didn't survive
        # compression, and checking it against the compressed sliver alone
        # produced false rejections (a textbook-correct "utmost good faith"
        # point, verbatim-supported by the retrieved chunk, scored 3/10
        # word matches against the compressed ~100-char version of that
        # same chunk and got dropped). Grounding checks should ask "is this
        # actually in what we retrieved," not "does it survive a budget
        # cut made for an unrelated reason."
        _full_context_uncompressed = "\n\n".join(c.page_content for c in all_chunks)

        total_retrieved_chars = sum(len(c.page_content) for c in all_chunks)
        if total_retrieved_chars > _context_budget:
            all_chunks = self._compressor.compress_to_budget(
                retrieval_query, all_chunks, max_total_chars=_context_budget
            )
        # Re-apply after compression, which may not preserve the ordering
        # from the earlier call — the generation prompt should see
        # topic-specific content first too, not just the grounding check.
        all_chunks = _prioritize_topic_chunks(retrieval_query, all_chunks)

        _VIDEO_SOURCE_TYPES = {"video", "youtube_transcript", "youtube"}
        _WEBPAGE_SOURCE_TYPES = {"webpage", "web"}
        context_parts, sources = [], []
        for chunk in all_chunks:
            source_type = chunk.metadata.get("source_type", "document")
            doc_type = chunk.metadata.get("doc_type", "")
            if source_type in _VIDEO_SOURCE_TYPES or doc_type == "youtube":
                url = chunk.metadata.get("source_url") or chunk.metadata.get("source", "Unknown URL")
                title = chunk.metadata.get("video_title", "")
                label = f"Video: {title or url}"
                sources.append(url)
            elif source_type in _WEBPAGE_SOURCE_TYPES:
                url = chunk.metadata.get("source_url") or chunk.metadata.get("source", "Unknown URL")
                label = f"Webpage: {url}"
                sources.append(url)
            else:
                src = chunk.metadata.get("source", "Unknown")
                page = chunk.metadata.get("page", "?")
                label = f"Document: {src} (Page {page})"
                sources.append(f"{src} (page {page})")
            # Phase 1 (contamination plan): generation used to see the
            # chunk's ENTIRE page_content, so a mixed "general"-tagged
            # chunk's off-topic sentence (e.g. a marine example embedded in
            # a motor-underwriting chunk) sat right there in the prompt and
            # the model dutifully turned it into its own point. Windowing
            # to the query-relevant excerpt(s) — the exact mechanism
            # _build_grounding_context already uses for the grounding check
            # — removes that sentence from what generation ever sees,
            # without excluding the chunk itself (its on-topic window still
            # contributes). _rerank_windows already returns the full text
            # unchanged for a chunk short enough that windowing wouldn't
            # help (<=700 chars, see its own docstring) — no separate
            # degenerate-chunk special-case needed here.
            _gen_windows = _rerank_windows(chunk.page_content, retrieval_query)
            context_parts.append(f"[{label}]\n" + "\n".join(_gen_windows))

        full_context = "\n\n".join(context_parts)

        # Prepend semantically related prior Q&A as supplementary context.
        # These are cached answers for related (but not identical) questions —
        # they enrich the context the LLM sees without replacing fresh retrieval.
        if _kv_related_ctx and full_context.strip():
            full_context = (
                "[Related prior answers — use as supporting context]\n"
                + _kv_related_ctx
                + "\n\n[Knowledge base chunks]\n"
                + full_context
            )

        # Fold in the prior-answer context detected earlier — either a literal
        # paste from the user, or a specific point pulled from the previous
        # numbered answer in history. When it alone already grounds the
        # answer (_pasted_grounds_answer), use it as the sole context — fresh
        # retrieval still ran (for the gates above and in case it's needed),
        # but its output would only dilute an already-sufficient answer with
        # unrelated KB noise. When it isn't enough on its own, keep both:
        # this context plus whatever fresh KB content retrieval found.
        #
        # EXCEPT for a point-reference follow-up ("explain point 3 in more
        # detail"): the point's own text trivially "grounds" a question
        # about itself — asking to explain X using a sentence about X always
        # passes _verify_grounding — so this branch's sole-context shortcut
        # left the model with literally nothing but that one sentence to
        # work from. Confirmed live: "explain point 3 in more detail" came
        # back as a near-verbatim reword of the original point ("covers the
        # entire lifecycle... from planning and financing to testing and
        # commissioning" -> "covers the entire journey... starting from
        # planning and financing right through to testing and
        # commissioning") — a paraphrase, not the additional depth the user
        # actually asked for. A genuine user-pasted block is different: the
        # user deliberately supplied that exact text as the thing to answer
        # from, so restricting to it is correct there. For a point
        # reference, always keep fresh retrieval alongside it so there's
        # real additional material to expand into when the KB has more to
        # say about that point's specific sub-topic.
        if pasted_context:
            if _pasted_grounds_answer and not _pasted_is_point_reference:
                full_context = "[Context from an earlier answer]\n" + pasted_context
                # The answer draws only from this, not from retrieval —
                # citing KB documents that were fetched but never actually
                # used would be a misleading source list.
                sources = []
            elif _pasted_is_point_reference:
                # Use the point-targeted retrieval from above, not the main
                # pipeline's full_context — that was retrieved using the
                # vague follow-up phrasing ("explain point 3 in more
                # detail") and can rank unrelated KB content ahead of
                # anything about the point's actual subject matter.
                full_context = (
                    "[Context from an earlier answer]\n"
                    + pasted_context
                    + ("\n\n[Related knowledge base content]\n" + _point_ref_context if _point_ref_context else "")
                )
                if _point_ref_sources:
                    sources = _point_ref_sources
            else:
                full_context = (
                    "[Context from an earlier answer]\n"
                    + pasted_context
                    + ("\n\n[Knowledge base chunks]\n" + full_context if full_context.strip() else "")
                )

        if not full_context.strip():
            yield (
                "Hmm, I don't have that specific information in my knowledge base right now. "
                "Let me get one of our agents on it, they'll be able to help you better! 😊"
            )
            yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": True})
            return
        else:
            # Use reformulated query as LLM question for detected follow-ups
            # ("give me in detail" → retrieval_query = "life insurance coverage details")
            # so the model sees the actual topic, not the vague follow-up phrase.
            # _detected_as_followup is used (not _is_followup) so the fix applies
            # even when we fell back to history-based topic extraction.
            #
            # A point-reference ("explain point 2 with an example") is a
            # SPECIAL case of this: retrieval_query comes from an LLM
            # reformulation step that's supposed to resolve "point 2" into
            # the point's actual subject, but that's exactly the kind of
            # instruction this model doesn't reliably follow (same lesson
            # as every other formatting-compliance gap fixed this session).
            # pasted_context, by contrast, is the point's exact text pulled
            # out by deterministic regex, not an LLM call — anchoring
            # prompt_question directly to it guarantees the model is always
            # told precisely what "point 2" IS, instead of trusting the
            # reformulation to have carried that meaning through correctly.
            # Confirmed live: without this, "explain point 2 with an
            # example" (point 2 being "also referred to as Medical
            # Insurance or Mediclaim") came back with a generic health-
            # insurance hospitalization example — a real example, just not
            # of what was actually asked about, because prompt_question
            # never explicitly said what point 2 was.
            if _pasted_is_point_reference and pasted_context:
                prompt_question = f'Explain this specific point: "{pasted_context}"'
            else:
                prompt_question = retrieval_query if _detected_as_followup else question

            # The fallback query-cleaning above may have dropped a specific
            # detail (a country/city name, "affordable"/"cheap"/"best") to
            # get a match at all — tell the model explicitly so it answers
            # with the general content it has while being honest that this
            # detail specifically isn't covered, rather than silently
            # presenting general info as if it addressed exactly what was
            # asked (or ignoring that the user asked about it at all).
            if _dropped_terms_note:
                prompt_question = (
                    f"{prompt_question.rstrip(' .?')}. Note: this knowledge base does not have "
                    f"specific information about {_dropped_terms_note} — explicitly say that up front, "
                    f"then answer using the general information you do have."
                )

            # ── Standalone-retry hedge phrasing ────────────────────────────
            # When the standalone retry saved the answer (the question was
            # NOT actually a follow-up to the prior topic), acknowledge the
            # ambiguity so the model doesn't confidently pretend it knew all
            # along by opening with something like "If you're asking about
            # {topic} generally, ..." (the model fills in the actual topic
            # name itself).
            if _answered_via_standalone_retry:
                prompt_question = (
                    f"{prompt_question.rstrip(' .?')}. This may be a new question "
                    f"unrelated to the earlier conversation; if so, open your answer "
                    f"with something like \"If you're asking about {{topic}} generally, \" "
                    f"before answering, naming the actual topic instead of the placeholder."
                )

            # ── Modifier-signal instruction injection ─────────────────────────
            # Detect what kind of modifier the user asked for (example / simple /
            # detail) and inject a targeted instruction into prompt_question so
            # the LLM knows exactly what format/style is expected. Reuses the
            # already-resolved intent (fast-path or LLM-fallback) from above
            # rather than independently re-running the fast-path-only checks.
            _has_example = _resolved_has_example
            _has_simple  = _resolved_has_simple
            _has_detail  = _resolved_has_detail

            if _detected_as_followup and (_has_example or _has_simple or _has_detail):
                # Build instruction based on the combination of signals.
                if _has_detail and _has_simple and _has_example:
                    _mod_instr = (
                        "Give a detailed explanation in simple, everyday language with one "
                        "concrete real-life example. No jargon. Explain like you would to a friend. "
                        "Use numbered points."
                    )
                    detailed = True  # SIMPLE normally overrides; force detailed prompt here
                elif _has_detail and _has_simple:
                    _mod_instr = (
                        "Give a thorough, detailed explanation using simple, everyday language. "
                        "No jargon or technical terms. Use numbered points."
                    )
                    detailed = True
                elif _has_detail and _has_example:
                    _mod_instr = (
                        "Give a full, detailed breakdown with a concrete real-life example."
                    )
                elif _has_simple and _has_example:
                    _mod_instr = (
                        "Re-explain in very simple, everyday language. No jargon. "
                        "Then give one concrete real-life example to illustrate it clearly."
                    )
                elif _has_example:
                    if _pasted_is_point_reference:
                        # A point reference needs a stronger anchor than the
                        # general case below — confirmed live: even with
                        # prompt_question explicitly quoting the point's
                        # exact text ('Explain this specific point: "It is
                        # also known as Medical Insurance or Mediclaim."'),
                        # the model still drifted to a generic hospital-
                        # coverage example instead of illustrating THAT
                        # specific fact. Some points (naming facts,
                        # definitional details) don't have a natural
                        # real-world scenario at all — forcing "a concrete
                        # real-life example" onto "it's also called X" was
                        # part of what pushed the model toward inventing an
                        # unrelated scenario instead. Giving it permission to
                        # clarify/illustrate the fact directly when a
                        # scenario doesn't fit keeps the answer on-topic
                        # either way.
                        _mod_instr = (
                            "Give ONLY an example or illustration of THIS EXACT POINT quoted "
                            "above, not a general example of the broader topic. If this point "
                            "is a naming fact, a definition detail, or something else without a "
                            "natural real-world scenario, clarify or illustrate that specific "
                            "fact directly instead of inventing an unrelated scenario. "
                            "Do NOT re-explain or repeat the definition first."
                        )
                    else:
                        _mod_instr = (
                            "The user already understands this concept from the previous answer. "
                            "Do NOT re-explain or repeat the definition. "
                            "Give ONLY one concrete real-life example that illustrates it clearly."
                        )
                elif _has_simple:
                    _mod_instr = (
                        "Re-explain this in very simple, everyday language. "
                        "No jargon or technical terms. Plain words only."
                    )
                elif _has_detail:
                    _mod_instr = (
                        "Give a full, detailed breakdown. "
                        "The user wants more depth. Use numbered points."
                    )
                else:
                    _mod_instr = ""
                if _mod_instr:
                    prompt_question = f"{prompt_question.rstrip(' .?')}. {_mod_instr}"
            elif not _detected_as_followup and _has_example:
                # Fresh question asking for an explanation with an example
                prompt_question = (
                    f"{prompt_question.rstrip(' .?')}. "
                    "Please include a concrete real-life example to illustrate this concept clearly."
                )

            if document_filter:
                prompt = STRICT_GROUNDED_PROMPT.format(history=history, context=full_context, question=prompt_question)
                llm = get_insurance_llm(temperature=0)
            elif detailed:
                prompt = DETAILED_GROUNDED_PROMPT.format(history=history, context=full_context, question=prompt_question)
                llm = get_insurance_llm(temperature=0)
            else:
                prompt = CONVERSATIONAL_RAG_PROMPT.format(
                    history=history,
                    context=full_context,
                    question=prompt_question,
                )
                llm = get_insurance_llm(temperature=0)

        # ── Stream LLM tokens directly via HTTP SSE ───────────────────────────
        # LangChain's astream() buffers the full response before yielding.
        # We bypass it and call the backend's /v1/chat/completions endpoint
        # directly with stream=True so the frontend sees the first word in
        # <1 second. Groq's API is OpenAI-compatible — identical SSE shape
        # to vLLM's — so both backends share this one streaming path rather
        # than Groq falling back to a single blocking response.
        import json as _json
        import aiohttp
        from router import VLLM_HOST, VLLM_API_KEY, _resolve_vllm_model, _active_backend

        unique_sources = list(dict.fromkeys(sources))
        streamed_ok = False
        _kv_reply = ""  # buffer for cache write
        # Captured from the SSE stream's finish_reason field (see the
        # streaming loop below) so the sentence cap further down can tell a
        # naturally-completed answer ('stop') from one that was cut off or
        # came from a path that doesn't report it (None) — see the "Hard
        # sentence cap" section for why this matters. Stays None on the
        # buffered/non-streaming fallback path, which the cap treats the
        # same conservative way it always has.
        _finish_reason = None

        # Streaming is always attempted below — always-stream, not gated on
        # _top_rerank. A prior version skipped live streaming when
        # _top_rerank < 0.05, reasoning that a low-confidence answer might
        # get a Rule 4 marker stripped afterward and the user shouldn't see
        # the raw pre-strip text flash by. That gate had a false-positive
        # problem specific to detailed answers: _top_rerank is the score of
        # the SINGLE best-matching chunk, but detailed mode deliberately
        # retrieves from a wider pool (_doc_top_k=14 vs 8) and builds
        # its context from many chunks combined — so the top single chunk
        # can legitimately score low even when the combined context is rich
        # and the generated answer is fully correct. Confirmed live: "can
        # you explain it in detail" after "What is motor insurance?" scored
        # _top_rerank=0.003 (gate would block streaming) yet produced a
        # complete, accurate, well-grounded answer citing the Motor
        # Vehicles Act — the gate was blocking exactly the case it was
        # never meant to catch, forcing every detailed answer through the
        # blocking, buffer-then-dump `llm.invoke()` fallback below (all-at-
        # once after the full ~600-token generation, instead of live).
        # Removing the gate doesn't remove the safety property it existed
        # for: the Rule4-strip/truncation-fix correction below still runs
        # unconditionally on the accumulated text regardless of whether it
        # was streamed, and the frontend already does a full atomic replace
        # of the displayed message when a corrected_text arrives (see
        # App.tsx) — so a rare low-confidence live-streamed answer that
        # needs correcting still gets fixed, exactly as before, it just
        # streams live first instead of staying invisible for the whole
        # generation.
        _active = _active_backend()
        _stream_url = None
        _stream_model = None
        _stream_headers = None
        if _active == "vllm" and VLLM_HOST:
            _stream_model = _resolve_vllm_model()
            _stream_url = f"{VLLM_HOST}/v1/chat/completions"
            _stream_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VLLM_API_KEY}",
            }
        elif _active == "groq":
            from router import GROQ_API_KEY, GROQ_MODEL
            _stream_model = GROQ_MODEL
            _stream_url = "https://api.groq.com/openai/v1/chat/completions"
            _stream_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            }
        elif _active == "manual":
            from router import _runtime_manual_api_key, _runtime_manual_base_url, _runtime_manual_model
            _stream_model = _runtime_manual_model
            _stream_url = f"{_runtime_manual_base_url.rstrip('/')}/chat/completions"
            _stream_headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {_runtime_manual_api_key}",
            }

        _t_promptbuild_ms = round((time.time() - _t_promptbuild_start) * 1000)
        _t_llm_start = time.time()
        if _stream_url:
            model = _stream_model
            url = _stream_url
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                # 600 was cutting genuine 8-point detailed answers off
                # mid-generation — an opener + 8 points + closer commonly
                # runs 450-550 tokens on its own, before accounting for
                # markdown bolding/bullets the model sometimes adds, which
                # left little headroom. Confirmed live: a fire-insurance
                # "explain in detail" answer was cut after ~2 points.
                "max_tokens": int(__import__("os").getenv("VLLM_MAX_TOKENS", "900") if detailed else __import__("os").getenv("VLLM_MAX_TOKENS_BRIEF", "300")),
                "stream": True,
            }
            headers = _stream_headers
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        # An HTTP error (rate limit, auth failure, server error,
                        # etc.) returns a JSON error body, not SSE "data: ..."
                        # lines — every line silently fails the startswith
                        # check below and gets skipped, so the loop finishes
                        # having yielded zero tokens while streamed_ok=True
                        # still gets set at the end, producing a SILENT EMPTY
                        # ANSWER with no error surfaced anywhere. Explicitly
                        # failing here instead routes into the except block
                        # below, which falls back to the non-streaming path
                        # (or, if that also fails, at least logs a real
                        # warning instead of nothing at all). Found via a
                        # genuine Groq 429 (daily token quota hit) that
                        # produced exactly this silent-empty-response bug.
                        if resp.status != 200:
                            body = await resp.text()
                            raise RuntimeError(f"HTTP {resp.status} from {_active} backend: {body[:300]}")
                        # vLLM SSE chunks may contain multiple "data: ..." lines.
                        # Reading resp.content by chunk (default) silently drops
                        # events when json.loads sees multi-line text. Buffer and
                        # split by newline to handle any chunk boundary correctly.
                        buf = ""
                        done = False
                        async for raw_chunk in resp.content:
                            buf += raw_chunk.decode("utf-8", errors="replace")
                            while "\n" in buf:
                                raw_line, buf = buf.split("\n", 1)
                                line = raw_line.strip()
                                if not line.startswith("data:"):
                                    continue
                                data = line[5:].strip()
                                if data == "[DONE]":
                                    done = True
                                    break
                                try:
                                    _choice = _json.loads(data)["choices"][0]
                                    token = _choice["delta"].get("content", "") or ""
                                    if token:
                                        if _t_ttft_ms is None:
                                            _t_ttft_ms = round((time.time() - _t_request_start) * 1000)
                                        _kv_reply += token
                                        yield token
                                    _fr = _choice.get("finish_reason")
                                    if _fr:
                                        _finish_reason = _fr
                                except Exception:
                                    pass
                            if done:
                                break
                streamed_ok = True
            except Exception as exc:
                logger.warning("[ask_stream] direct %s streaming failed, falling back: %s", _active, exc)

        if not streamed_ok:
            # Fallback: regular invoke (no streaming) — only reached for
            # openai/anthropic backends, or if the vLLM/Groq streaming call
            # above raised an exception.  We buffer the full response and
            # apply all post-processing (Rule4 strip, truncation detection,
            # sentence cap) BEFORE yielding, so the client never sees raw
            # un-corrected text.
            response = await asyncio.to_thread(llm.invoke, prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            answer = _strip_markdown(_strip_model_preamble(answer))
            _kv_reply = answer
        _t_llm_ms = round((time.time() - _t_llm_start) * 1000)
        _t_postllm_start = time.time()

        # ── Rule 4 fallback strip ─────────────────────────────────────────────
        # The 7B model sometimes generates real content from training knowledge
        # and then ALSO appends the Rule 4 canned fallback at the end because
        # the specific fact wasn't in the retrieved context. When real content
        # precedes the fallback, strip the fallback so users only see the answer.
        # trust_content used to reuse _rerank_gate directly, on the reasoning
        # that "good enough to generate from is good enough to trust
        # post-hoc". That stopped being true once _rerank_gate was separately
        # lowered to 0.0005 (a near-zero noise backstop, not a relevance bar)
        # to fix an unrelated false-rejection problem — the two changes
        # combined made _r4_trusted almost always True, so the model's own
        # explicit "I don't have that specific info" hedge got silently
        # discarded even when the leading claim was genuinely ungrounded
        # (observed: "IRDAI guidelines specify certain obligations... honestly,
        # I don't have that specific info" — the hedge was right, but got
        # stripped anyway, leaving a confident-sounding vague claim). Use the
        # same 0.05 bar as the single-topic lexical-miss rescue elsewhere in
        # this file instead — below that, trust the model's own uncertainty
        # signal rather than overriding it with a rerank_gate that no longer
        # means "relevant enough", just "not pure noise".
        _reply_stripped = (_kv_reply or "").rstrip()
        _corrected_text = None

        # The prompt bans the filler word "honestly"/"honest," outright, but
        # the model doesn't reliably follow that (confirmed live, same call,
        # non-deterministic: "Honestly, it's..." on one run, "Honest, it's..."
        # on the next, identical question and context — this was already a
        # known compliance gap before the word was banned entirely). Same
        # lesson as the Rule4/truncation fixes right below: don't trust a
        # prompt instruction the model won't consistently honor — enforce it
        # deterministically instead. Strips the filler and capitalizes the
        # word that follows so the sentence still reads naturally.
        _honest_fixed = _HONESTLY_FILLER_RE.sub(lambda m: m.group(1).upper(), _reply_stripped)
        if _honest_fixed != _reply_stripped:
            _reply_stripped = _honest_fixed
            _kv_reply = _honest_fixed
            _corrected_text = _honest_fixed

        _unbolded = _MARKDOWN_BOLD_RE.sub(r"\1", _reply_stripped)
        if _unbolded != _reply_stripped:
            _reply_stripped = _unbolded
            _kv_reply = _unbolded
            _corrected_text = _unbolded

        _no_emdash = _EM_DASH_RE.sub(", ", _reply_stripped)
        _no_emdash = re.sub(r",\s*([.!?])", r"\1", _no_emdash)
        _no_emdash = re.sub(r",\s*,", ",", _no_emdash)
        if _no_emdash != _reply_stripped:
            _reply_stripped = _no_emdash
            _kv_reply = _no_emdash
            _corrected_text = _no_emdash

        # See _NON_LATIN_SCRIPT_RE above: rare vLLM language-leakage
        # artifact where the model tacks on a CJK-script tail (or, more
        # rarely, a mid-sentence run) after an otherwise-English answer.
        # Gated on an explicit .search() first — confirmed live this was
        # a real bug, not just a defensive habit: the unconditional
        # `re.sub(r"\s+", " ", ...)` cleanup ran on every reply
        # regardless of whether anything matched, collapsing the "\n\n"
        # before numbered-list items into a single space even on
        # perfectly clean English answers. That erased the newline the
        # downstream numbered-list-enforcement check relies on to
        # recognize an already-numbered list, which misfired and shredded
        # every "N. " marker into its own fragment — corrupting detailed
        # answers (health/claim-steps) that had nothing to do with
        # language leakage at all. Only touch the text, and only collapse
        # whitespace, when the CJK regex actually found something.
        # Replace each matched run with a single space rather than
        # deleting it outright, so words on either side of a mid-string
        # occurrence don't get jammed together. Horizontal whitespace
        # (spaces/tabs) only — newlines are left alone so list/paragraph
        # structure elsewhere in the reply survives untouched. Only apply
        # the fix if it preserves at least 40% of the original content —
        # if more than that was non-Latin script, the generation is
        # corrupted beyond a simple trim and it's safer to leave the text
        # untouched (and visible for debugging) than to silently emit a
        # near-empty answer.
        if _NON_LATIN_SCRIPT_RE.search(_reply_stripped):
            _relatinized = _NON_LATIN_SCRIPT_RE.sub(" ", _reply_stripped)
            _relatinized = re.sub(r"[ \t]+", " ", _relatinized)
            _relatinized = re.sub(r"[ \t]*\n[ \t]*", "\n", _relatinized).strip()
            _relatinized = re.sub(r"[\s,;:\-–—]+$", "", _relatinized)
            if _relatinized and len(_relatinized) >= 0.4 * len(_reply_stripped):
                if _relatinized[-1].isalnum():
                    _relatinized += "."
                logger.warning(
                    "[ask_stream] stripped non-Latin-script text from reply (%d -> %d chars) — vLLM language-leakage artifact",
                    len(_reply_stripped), len(_relatinized),
                )
                _reply_stripped = _relatinized
                _kv_reply = _relatinized
                _corrected_text = _relatinized

        _rule4_discarded = False
        _r4_trusted = _top_rerank >= 0.05
        _r4_stripped = _strip_rule4_fallback(_reply_stripped, trust_content=_r4_trusted)
        if _r4_stripped:
            _reply_stripped = _r4_stripped
            _kv_reply = _r4_stripped
            _corrected_text = _r4_stripped
            logger.info("[ask_stream] stripped Rule4 fallback appended after real content (%d chars)", len(_r4_stripped))
        elif _r4_stripped == "":
            # Marker present but the leading content didn't even clear the
            # entry gate (shouldn't normally happen since generation only
            # runs on chunks that passed it) — discard and show the
            # standard refusal instead of an unconfirmed claim.
            _refusal_text = (
                "Hmm, I don't have that specific information in my knowledge base right now. "
                "Let me get one of our agents on it, they'll be able to help you better! 😊"
            )
            _reply_stripped = _refusal_text
            _kv_reply = _refusal_text
            _corrected_text = _refusal_text
            _rule4_discarded = True
            logger.info(
                "[ask_stream] discarded low-confidence Rule4 answer (top_rerank=%.3f < 0.05)",
                _top_rerank,
            )

        # ── Truncation detection (log only, never discard content) ────────────
        # Used to trim the answer down to its last complete sentence when the
        # stream hit max_tokens mid-sentence. Explicitly disabled per user
        # instruction: whatever was actually generated should reach the user,
        # not be cut down further — the trim regularly threw away most of a
        # detailed answer (an 8-point answer could end up as 2 sentences)
        # whenever the real fix is raising max_tokens (done above) or
        # investigating why generation stopped early, not hiding the result.
        # Also mis-fired on ordinary complete answers ending in a trailing
        # emoji ("...this. 😊") since "😊" isn't a sentence-ending character —
        # kept only as a log signal for how often that's still happening.
        _SENT_ENDERS = frozenset('.!?…')
        if _reply_stripped and _reply_stripped[-1] not in _SENT_ENDERS:
            logger.info(
                "[ask_stream] answer does not end at a sentence boundary (len=%d) — showing as generated",
                len(_reply_stripped),
            )

        # ── Numbered-list enforcement (detailed mode) ─────────────────────────
        # DETAILED_GROUNDED_PROMPT explicitly instructs "1. ... 2. ... 3. ..."
        # numbered points with worked right/wrong examples — the model doesn't
        # reliably follow it (confirmed live: a well-grounded, factually
        # correct detailed answer about motor insurance came back as one
        # unbroken paragraph despite the prompt's explicit numbering
        # instruction and examples). Same lesson as every other formatting-
        # compliance gap fixed this session (the "honestly"->"honest" filler
        # word, Rule4 markers, truncation) — don't trust a prompt instruction
        # this model won't consistently honor, enforce it deterministically
        # after the fact instead. Only fires when the answer has NO numbered
        # points at all (a real "1. " or "\n1." pattern) so an already-correct
        # numbered response is left untouched. Splits on sentence boundaries,
        # treats the first sentence as the warm opener the prompt asks for
        # (left unnumbered), the closing farewell line as unnumbered too, and
        # numbers everything in between — mirroring the prompt's own intended
        # structure rather than inventing a different one.
        if _keyword_detailed:
            try:
                import re as _re
                _num_src = (_corrected_text or _reply_stripped).strip()
                _already_numbered = bool(_re.search(r'(?:^|\n)\s*1\.\s', _num_src))
                if _num_src and not _already_numbered:
                    # (?<!\d\.) — same fix as the sentence cap below: don't
                    # treat a bare numbered-list marker as its own sentence
                    # boundary. Less likely to bite here since this branch
                    # only runs when the answer has NO numbering yet, but a
                    # stray "Section 5. of the Act..." or similar could still
                    # trip the naive version, so keep both occurrences
                    # consistent.
                    _sentences = [s.strip() for s in _re.split(r'(?<=[.!?])(?<!\d\.)\s+', _num_src) if s.strip()]
                    _CLOSING_RE = _re.compile(
                        r'^(hope that|let me know|feel free|hang tight|glad to)', _re.IGNORECASE,
                    )
                    _closing_parts = []
                    while len(_sentences) > 1 and _CLOSING_RE.match(_sentences[-1]):
                        _closing_parts.insert(0, _sentences.pop())
                    _closing = " ".join(_closing_parts)
                    # Need at least 2 sentences left after removing the opener
                    # to make numbering worthwhile — a 1-2 sentence answer is
                    # already effectively "one point", forcing "1. " on it
                    # would look broken rather than helpful.
                    if len(_sentences) >= 3:
                        _opener = _sentences.pop(0)
                        _points = "\n".join(f"{i}. {s}" for i, s in enumerate(_sentences, 1))
                        _rebuilt = _opener + "\n\n" + _points
                        if _closing:
                            _rebuilt += "\n\n" + _closing
                        _corrected_text = _rebuilt
                        _kv_reply = _rebuilt
                elif _re.match(r'^\s*1\.\s', _num_src):
                    # The model numbers correctly but sometimes skips the
                    # warm opener FORMAT also asks for — confirmed live: a
                    # fresh "Explain X in detail" query (not a follow-up)
                    # reliably returned "1. Fire insurance is a type of..."
                    # / "1. Travel insurance offers..." with nothing before
                    # the numbering, 2/2 on different topics, while the same
                    # request phrased as a follow-up ("Can you explain it in
                    # detail.") reliably included an opener. Same lesson as
                    # the block above: don't trust prompt-only compliance
                    # this model treats as conditional on phrasing.
                    _rebuilt = f"Sure, here's a detailed breakdown:\n\n{_num_src}"
                    _corrected_text = _rebuilt
                    _kv_reply = _rebuilt
            except Exception as _num_exc:
                logger.debug("[ask_stream] numbered-list enforcement skipped: %s", _num_exc)

        # ── Drop ungrounded padding points (detailed mode) ────────────────────
        # DETAILED_GROUNDED_PROMPT already says "cover only the points the
        # KNOWLEDGE BASE actually makes... every added sentence is another
        # chance to say something unsupported" and "never pad or invent to
        # reach 8" — the model doesn't reliably follow it. Confirmed live on a
        # genuinely well-grounded fire insurance answer: 5 of 8 points were
        # near-verbatim KB text, but 3 were generic wrap-up filler ("makes it
        # a versatile option for property owners", "can be tailored to meet
        # the specific needs...", "will help you choose the right policy...
        # effectively") that restated nothing specific from the source, just
        # padding to make the list look longer/more complete.
        #
        # Enforced deterministically: a point survives only if it shares
        # enough substantial (4-word) phrases with the actual retrieved
        # context — genuine KB content, even paraphrased, keeps some exact
        # source wording; invented summary sentences don't. Uses an absolute
        # match count, not a ratio, because a long near-verbatim point
        # naturally has a lower MATCHED/TOTAL fraction than a short
        # half-invented one simply from length — confirmed live: the
        # genuinely-grounded points scored 11-31 matched 4-grams each, while
        # the 3 padding points scored 0-4, a clean, wide gap. A point that
        # opens with a real KB phrase then pivots to an invented claim (the
        # "versatile option" case: matched=4, borderline) is intentionally
        # held to the same bar as pure invention — partial grounding for one
        # clause doesn't excuse fabricating the rest of the sentence.
        # A shorter, fully-grounded list beats a longer one with invented
        # filler — matches the user's explicit "fewer points is fine, just
        # don't invent" instruction.
        # Default so the contamination trace below can safely reference
        # this even if _keyword_detailed is False or this block's try
        # raises before reaching the assignment.
        _trace_original_points: list = []
        if _keyword_detailed:
            try:
                import re as _re3
                _filter_src = (_corrected_text or _reply_stripped).strip()
                # Shared with the cross-topic filter below, the Phase-2
                # relevance gate, and the contamination trace — see
                # _split_numbered_points's own docstring for why this used
                # to be duplicated inline here.
                _opener_lines, _point_texts, _closer_lines = _split_numbered_points(_filter_src)
                # Original, pre-filter point set — captured here (not
                # recomputed later) so the contamination trace below can
                # show every point that was actually generated, including
                # ones this filter or the cross-topic filter go on to drop.
                _trace_original_points = list(_point_texts)
                if len(_point_texts) >= 2 and _full_context_uncompressed and _full_context_uncompressed.strip():
                    # Must strip punctuation the same way on both sides.
                    # full_context is raw KB prose, dense with commas
                    # ("lightning, explosion, aircraft damage, riot..."),
                    # while _matched_words() below extracts \w+ words from
                    # each point. This class of bug (asymmetric normalization
                    # between the two strings being compared) generalizes
                    # beyond this one spot — keep both sides passing through
                    # the exact same tokenizer in any future overlap/
                    # grounding check.
                    # Light plural stemming, applied identically to both
                    # sides (same principle as the punctuation handling above
                    # — any asymmetry between the two strings being compared
                    # creates false negatives). Confirmed live, back when
                    # this check still matched exact 4-word phrases rather
                    # than plain words: the model pluralized an entire near-
                    # verbatim peril-list point ("explosion"->"explosions",
                    # "riot"->"riots", "storm"->"storms", etc. throughout),
                    # and matching against the KB's singular forms scored 0
                    # matches out of 16 possible — the point was dropped
                    # despite being genuine KB content, just re-pluralized.
                    # Kept even after switching to word-level overlap (see
                    # _matched_words below): plain word overlap is far more
                    # forgiving of reordering than phrase matching was, but
                    # an un-stemmed "explosions" still wouldn't equal a
                    # singular "explosion" token without this.
                    def _norm_word(w: str) -> str:
                        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
                            return w[:-1]
                        return w

                    def _norm_words(text: str) -> list[str]:
                        return [_norm_word(w) for w in _re3.findall(r"\w+", text.lower())]

                    _ctx_word_set = set(_norm_words(_full_context_uncompressed))
                    # Switched from exact 4-word phrase matching to plain
                    # significant-word overlap after the phrase-based check
                    # proved fundamentally too brittle to fix by threshold
                    # tuning alone — confirmed live (2026-07-10) with paired
                    # measurements on the same real answer: one point scored
                    # 2/12 matched 4-word phrases (would fail even a loosened
                    # bar) but 9/9 (100%) of its individual significant words
                    # appeared in the context — every word was genuine, just
                    # not in the same 4-word sequences as the source, which
                    # is exactly what normal paraphrasing does. Every point
                    # in that test scored 62-100% word overlap regardless of
                    # phrase-match score, meaning word overlap correctly
                    # recognized all of them as grounded while phrase
                    # matching arbitrarily rejected most. Order/adjacency-
                    # sensitive matching was never the right test for "did
                    # this paraphrase come from the source" — a sentence can
                    # rearrange every clause and still say only what the KB
                    # says.
                    # Threshold picked from the same measurement: confirmed-
                    # genuine points scored 62-100%, so 50% leaves real
                    # margin below the observed floor while still requiring
                    # a point draw the clear majority of its content words
                    # from the actual retrieved text — a point built from
                    # invented specifics (numbers, product names, claims not
                    # in the source) won't clear that bar just by reusing
                    # generic insurance vocabulary that happens to appear in
                    # any context on the topic.
                    _MIN_MATCHES = 4
                    _MIN_FRACTION = 0.5

                    def _matched_words(text: str) -> tuple[int, int]:
                        _words = _norm_words(text)
                        _content = {w for w in _words if len(w) >= 4}
                        if not _content:
                            return 0, 0
                        _matched = sum(1 for w in _content if w in _ctx_word_set)
                        return _matched, len(_content)

                    def _point_grounded(text: str) -> bool:
                        _matched, _total = _matched_words(text)
                        if _total <= 0:
                            return False
                        # ceil(total * 0.5) via integer math
                        _frac_threshold = -(-_total // 2)
                        _threshold = min(_MIN_MATCHES, max(1, _frac_threshold))
                        return _matched >= _threshold

                    _kept_points = [p for p in _point_texts if _point_grounded(p)]
                    if _kept_points and len(_kept_points) < len(_point_texts):
                        _rebuilt3 = _rebuild_from_points(_opener_lines, _kept_points, _closer_lines)
                        _corrected_text = _rebuilt3
                        _kv_reply = _rebuilt3
                        logger.info(
                            "[ask_stream] dropped %d ungrounded point(s) from detailed answer",
                            len(_point_texts) - len(_kept_points),
                        )
            except Exception as _filter_exc:
                logger.debug("[ask_stream] ungrounded-point filter skipped: %s", _filter_exc)

        # ── Drop cross-topic-contaminated points (detailed mode) ──────────────
        # The padding-point filter above only catches points that AREN'T
        # grounded in full_context at all. It can't catch a point that IS
        # genuinely grounded — because a generic, legitimately-relevant chunk
        # (e.g. "how a policy schedule's terms & conditions are documented")
        # got retrieved alongside the topic-specific chunks, but that generic
        # chunk used a DIFFERENT insurance type as its own illustrative
        # example, and the model reproduced that example as if it described
        # the topic actually being asked about. Confirmed live: "Explain
        # motor insurance in detail" produced a point about "marine cargo
        # policies... whether the cover is under ICC (A) or ICC (B)" — real
        # KB text, just from a generic underwriting-practices chunk's own
        # marine-insurance example, not from anything about motor insurance.
        # A second point the same run stated motor insurance "typically
        # cover[s] expenses related to hospitalization or domiciliary
        # hospitalization for illnesses" — health-insurance phrasing
        # (confirmed: a health-insurance exclusions chunk was genuinely in
        # the retrieved pool for that request) misapplied to a vehicle policy.
        # Also confirmed on a different topic: "Explain engineering insurance
        # in detail" stated "Hull insurance is a component of marine
        # insurance" and, in a later retry, "Cargo insurance within marine
        # insurance covers..." — same failure shape, different specific
        # jargon and different attribution phrasing each time ("component
        # of"/"aspect of"/"within"). Matching the literal type name itself
        # ("marine insurance") rather than trying to enumerate every possible
        # attribution phrase is what actually generalizes across phrasing —
        # there's essentially no legitimate reason an answer about a
        # DIFFERENT topic would explicitly name "marine insurance" at all.
        #
        # Enforced with a small, high-confidence list of jargon terms that
        # are essentially unique to one insurance type — checked against
        # whether the QUERY itself also names that type (so a genuine
        # comparison answer, "how does X differ from Y", is never affected,
        # since both type names would legitimately appear in the query).
        if _keyword_detailed:
            try:
                import re as _re4
                # "fidelity insurance" itself added after a confirmed live
                # miss: "Explain burglary insurance in detail" pulled a
                # "Lesson Round Up" chapter-summary chunk (already partly
                # handled by _CHAPTER_REVIEW_RE below, which only
                # deprioritizes such chunks rather than excluding them) and
                # reproduced its fidelity-insurance bullet verbatim as a
                # numbered point about burglary insurance: "Fidelity
                # insurance protects organizations from loss of money,
                # securities, or inventory resulting from crime." That exact
                # sentence doesn't contain "employee dishonesty" or
                # "embezzlement" — those appear in the chunk's NEXT sentence,
                # which the model didn't reproduce — so the existing terms
                # missed it even though "fidelity" was already a tracked
                # type. The type name itself is the one giveaway phrase that
                # will always be present regardless of which specific
                # sentence from a fidelity-insurance chunk gets echoed,
                # matching how "marine insurance" is already used for the
                # marine case above.
                #
                # _TYPE_GIVEAWAY_TERMS / _TYPE_QUERY_EXEMPT_WORDS are now
                # module-level (see definition near _TYPE_ATTRIBUTION_RE) so
                # the brief-mode whole-reply check below shares the exact
                # same list instead of drifting out of sync.
                _query_lower = retrieval_query.lower()
                _contam_src = (_corrected_text or _reply_stripped).strip()
                # Shared with the ungrounded-point filter above — see
                # _split_numbered_points's docstring.
                _contam_opener, _contam_points, _contam_closer = _split_numbered_points(_contam_src)
                # This filter can run even when the ungrounded-point filter
                # above didn't fire (e.g. only 1 point total, or that block's
                # try/except skipped) — cover that case so the trace still
                # has an original point set to show.
                if not _trace_original_points:
                    _trace_original_points = list(_contam_points)

                def _is_contaminated(point_text: str) -> bool:
                    # Skip entirely for type-agnostic queries ("what's the
                    # difference between NCB and TPA") — there's no "wrong
                    # topic" to detect when the query never named a type in
                    # the first place, and TPA/NCB-style cross-cutting jargon
                    # legitimately requires naming a specific type (TPA is
                    # genuinely explained via health insurance) to answer
                    # correctly. Confirmed live: this exact query got
                    # discarded because a fully correct answer said "health
                    # insurance services" — reuses the same policy_type
                    # classification already computed for the metadata
                    # retrieval filter above, not a new heuristic.
                    if _query_policy_type == "general":
                        return False
                    return _text_has_giveaway_contamination(
                        point_text, _query_lower, _query_policy_type, _full_context_uncompressed
                    )

                _clean_points = [p for p in _contam_points if not _is_contaminated(p)]
                if _clean_points and len(_clean_points) < len(_contam_points):
                    _rebuilt4 = _rebuild_from_points(_contam_opener, _clean_points, _contam_closer)
                    _corrected_text = _rebuilt4
                    _kv_reply = _rebuilt4
                    logger.info(
                        "[ask_stream] dropped %d cross-topic-contaminated point(s) from detailed answer",
                        len(_contam_points) - len(_clean_points),
                    )
            except Exception as _contam_exc:
                logger.debug("[ask_stream] cross-topic contamination filter skipped: %s", _contam_exc)

        # ── Semantic per-point topic-relevance gate (Phase 2, plan.md) ─────────
        # LOG-ONLY by default (POINT_RELEVANCE_GATE_ACTIVE unset) — this is
        # the SECOND design, after the first (ratio-to-max) was confirmed
        # live to gut legitimate answers. See git history / memory
        # project_point_relevance_ratio_to_max_unsafe.md for that failure;
        # this design specifically fixes the root cause it exposed.
        #
        # The two filters above only catch what they were built to catch: a
        # point ungrounded in the retrieved text, or one naming a small,
        # hardcoded list of type-giveaway jargon. Neither asks the one
        # question that actually generalizes: "is this point about what the
        # user asked?" This gate does, via the shared cross-encoder
        # reranker already resident in-process (_score_points_against_query).
        #
        # ISOLATION/GAP design, not ratio-to-max. Anchoring to the single
        # highest-scoring point (the first attempt) is fragile whenever one
        # point closely echoes the query and scores far above the rest —
        # every other point then looks like an outlier by comparison, even
        # fully legitimate ones (a cross-encoder naturally scores a narrow,
        # correct, specific point LOW against a broad "explain X in detail"
        # query — that reflects topical narrowness, not contamination).
        #
        # Instead: sort this answer's own point scores, find the single
        # largest consecutive-ratio jump, and treat everything below that
        # jump as a candidate drop ONLY IF that low group is a genuine
        # MINORITY of the points (a real contaminated point is rare among
        # an otherwise-good answer, not most of it) AND the jump itself is
        # a real cliff, not gradual variation. Validated against the two
        # concrete cases already on hand: the confirmed real contamination
        # (7 points, scores 0.0001-0.538) has its largest gap (57.8x)
        # isolate just 1 of 7 points (14%) — a minority, correctly
        # droppable. The confirmed false positive (8 points, scores
        # 0.0018-0.9634) has its largest gap (19.27x) isolate 6 of 8 points
        # (75%) — a MAJORITY — correctly rejected by the minority
        # constraint alone, even though the raw gap ratio is comparably
        # large in both cases. The minority constraint, not the gap size,
        # is what separates the two — gap size alone is not sufficient.
        #
        # 2026-07-22: activated once, immediately reverted. The isolation/
        # gap math above is necessary but not sufficient — a full-corpus
        # retest with per-run points_dropped instrumentation (added to
        # contamination_corpus_runner.py) found it deleted 8 legitimate
        # points across 4 different clean-control answers while the one
        # real, already-known leak (motor jargon in a personal-accident
        # answer) survived unchanged. Root cause: the cross-encoder scores
        # a point against the QUERY string, and "explain X in detail"
        # queries make every point in the answer score low (0.001-0.44,
        # nothing confident) — the isolation math then faithfully finds a
        # gap *inside that noise* and deletes whichever point is lowest,
        # which is just as often a correct, narrow sub-detail ("Assignment
        # differs from Nomination" in a life-insurance answer) as it is
        # real contamination. No absolute score ceiling fixes this: real
        # contamination and the false-positive deletions score in the
        # same ~0.001-0.05 band. See contamination_gate_active_phase2.json
        # for the full data.
        #
        # Two independent guards added below, neither of which existed in
        # the reverted version:
        #   1. CONFIDENCE FLOOR — only even attempt isolation when at
        #      least one point in the answer scores confidently on-topic
        #      (>= 0.5). If the model can't confidently place its OWN best
        #      point, the internal gaps between the rest are noise, not
        #      signal — this alone rules out every false positive seen in
        #      the retest (their maxima were 0.28-0.44).
        #   2. FOREIGN-TYPE CONFIRMATION — a candidate only survives if ITS
        #      OWN TEXT classifies (via the existing classify_query_
        #      policy_type, already used for retrieval filtering) as a
        #      real, SPECIFIC type that differs from this query's own
        #      type. A narrow-but-correct sub-detail classifies "general"
        #      or matches the query's type and is spared; genuine cross-
        #      topic content (a marine ICC clause, a motor-rider clause)
        #      classifies as that foreign type and is caught. Tested
        #      directly against the retest's actual points: every false
        #      positive classifies general/same-type (survives); the
        #      marine-in-transit true positive classifies "marine"
        #      (caught). Known gap, accepted: contamination laundered
        #      through a "general"-tagged source chunk (the crop-in-motor
        #      case) also classifies "general" and slips through this
        #      guard too — this stops being purely type-agnostic and
        #      leans on the type classifier's own blind spot. Not worse
        #      than before (that case wasn't caught by the reverted
        #      version either), but not the full fix.
        #
        # Action changed from DELETE to DEMOTE. Even with both guards, the
        # signal underneath is the same one that just produced real false
        # positives — deleting is irreversible and this session's finding
        # doesn't justify trusting it that far yet. Demoting confirmed
        # points to the end of the list costs nothing when the gate is
        # wrong (the content is still there, just deprioritized) and still
        # de-emphasizes genuine contamination when it's right.
        _POINT_MIN_GAP_RATIO = 5.0
        _POINT_MAX_MINORITY_FRACTION = 0.4
        _POINT_ISOLATED_ABS_CEILING = 0.15
        _POINT_CONFIDENCE_FLOOR = 0.5
        _POINT_GATE_ACTIVE = os.getenv("POINT_RELEVANCE_GATE_ACTIVE", "").strip().lower() in ("1", "true", "yes")
        # Demoted points get reordered, not removed, so original_point_count
        # == final_point_count even when this gate fires — the trace's
        # point-count fields can no longer show whether it acted. This
        # threads through which points (if any) were actually demoted this
        # request, so a corpus sweep can verify demotions land only on
        # confirmed-foreign content and never on a clean-control case.
        _pr_demoted: list = []
        if _keyword_detailed:
            try:
                _relevance_src = (_corrected_text or _reply_stripped).strip()
                _rel_opener, _rel_points, _rel_closer = _split_numbered_points(_relevance_src)
                if not _trace_original_points:
                    _trace_original_points = list(_rel_points)
                # Need at least 3 points: with a minority fraction of 0.4,
                # anything smaller can never isolate a non-empty minority
                # group in the first place.
                if len(_rel_points) >= 3:
                    _rel_scores = _score_points_against_query(retrieval_query, _rel_points)
                    if (
                        _rel_scores
                        and len(_rel_scores) == len(_rel_points)
                        and max(_rel_scores) >= _POINT_CONFIDENCE_FLOOR
                    ):
                        _n_points = len(_rel_points)
                        # Sort INDICES by score (not the point strings
                        # themselves) — unambiguous even if two points
                        # happen to share identical text.
                        _rank_order = sorted(range(_n_points), key=lambda i: _rel_scores[i])
                        _max_minority_size = int(_n_points * _POINT_MAX_MINORITY_FRACTION)
                        # Find the largest consecutive-ratio gap, but only
                        # among split points that would leave the LOW side a
                        # minority — a huge gap that splits off a majority
                        # (the false-positive shape) is never even a
                        # candidate, not just rejected after the fact.
                        _best_ratio, _split_at = 0.0, -1
                        for i in range(min(_max_minority_size, _n_points - 1)):
                            _cur = max(_rel_scores[_rank_order[i]], 1e-6)
                            _nxt = _rel_scores[_rank_order[i + 1]]
                            _ratio = _nxt / _cur
                            if _ratio > _best_ratio:
                                _best_ratio, _split_at = _ratio, i
                        _isolated_idx = _rank_order[:_split_at + 1] if _split_at >= 0 else []
                        _drop_idx = [i for i in _isolated_idx if _rel_scores[i] <= _POINT_ISOLATED_ABS_CEILING]
                        if (
                            _drop_idx
                            and len(_drop_idx) == len(_isolated_idx)
                            and _best_ratio >= _POINT_MIN_GAP_RATIO
                        ):
                            # Guard 2: foreign-type confirmation. classify
                            # each RAW candidate's own text; only points
                            # that name a real, different type survive
                            # into _confirmed_idx.
                            _confirmed_idx = []
                            for _di in _drop_idx:
                                try:
                                    _pt_type = classify_query_policy_type(_rel_points[_di])
                                except Exception:
                                    _pt_type = "general"
                                if _pt_type != "general" and _pt_type != _query_policy_type:
                                    _confirmed_idx.append(_di)
                            logger.info(
                                "[ask_stream] point-relevance gate CANDIDATE demote=%d/%d "
                                "confirmed=%d active=%s (gap=%.2fx, sorted_scores=%s, "
                                "query_type=%s)",
                                len(_drop_idx), _n_points, len(_confirmed_idx), _POINT_GATE_ACTIVE,
                                _best_ratio, [round(_rel_scores[i], 4) for i in _rank_order],
                                _query_policy_type,
                            )
                            if _POINT_GATE_ACTIVE and _confirmed_idx:
                                # Demote, don't delete: confirmed points move
                                # to the end in their original relative
                                # order; every point the model wrote is
                                # still in the answer.
                                _pr_demoted = [
                                    {"text": _rel_points[i][:200], "confirmed_type": classify_query_policy_type(_rel_points[i])}
                                    for i in _confirmed_idx
                                ]
                                _demote_set = set(_confirmed_idx)
                                _reordered_points = (
                                    [p for i, p in enumerate(_rel_points) if i not in _demote_set]
                                    + [_rel_points[i] for i in _confirmed_idx]
                                )
                                _rebuilt5 = _rebuild_from_points(_rel_opener, _reordered_points, _rel_closer)
                                _corrected_text = _rebuilt5
                                _kv_reply = _rebuilt5
            except Exception as _rel_exc:
                logger.debug("[ask_stream] point-relevance gate skipped: %s", _rel_exc)

        # ── Contamination trace (Phase 0, plan.md) ─────────────────────────────
        # Opt-in via CONTAMINATION_TRACE=1. Logs the FINAL surviving points
        # (after both filters above have already run) with a per-point
        # query-relevance score from the shared cross-encoder and a
        # best-guess source-chunk attribution — pure diagnostics, nothing
        # here changes the answer. This is what makes contamination
        # visible in bulk instead of one live repro at a time; see
        # contamination_trace.py's module docstring. _trace_original_points
        # is not yet used here (surviving == what the trace shows) but is
        # threaded through so a later stage can report per-point drop
        # attribution without re-deriving it.
        if _keyword_detailed and contamination_trace.TRACE_ENABLED:
            try:
                _trace_src = (_corrected_text or _reply_stripped).strip()
                _, _trace_final_points, _ = _split_numbered_points(_trace_src)
                if _trace_final_points:
                    _trace_scores = _score_points_against_query(retrieval_query, _trace_final_points)
                    _trace_chunk_texts = [getattr(c, "page_content", "") for c in all_chunks]
                    _trace_point_records = []
                    for _ti, _tpoint in enumerate(_trace_final_points):
                        _tscore = _trace_scores[_ti] if _ti < len(_trace_scores) else None
                        _tbest_idx = contamination_trace.best_matching_chunk_index(_tpoint, _trace_chunk_texts)
                        _tbest_chunk = all_chunks[_tbest_idx] if _tbest_idx is not None else None
                        _trace_point_records.append({
                            "text": _tpoint[:200],
                            "relevance_score": _tscore,
                            "best_chunk_source": (
                                _tbest_chunk.metadata.get("source") or _tbest_chunk.metadata.get("source_url")
                            ) if _tbest_chunk else None,
                            "best_chunk_policy_type": _tbest_chunk.metadata.get("policy_type") if _tbest_chunk else None,
                        })
                    contamination_trace.write_trace({
                        "query": question,
                        "retrieval_query": retrieval_query,
                        "query_policy_type": _query_policy_type,
                        "query_candidate_type": _query_candidate_type,
                        "original_point_count": len(_trace_original_points),
                        "final_point_count": len(_trace_final_points),
                        "point_relevance_demoted": _pr_demoted,
                        "retrieved_chunks": [
                            {
                                "source": c.metadata.get("source") or c.metadata.get("source_url"),
                                "policy_type": c.metadata.get("policy_type"),
                                "candidate_policy_type": c.metadata.get("candidate_policy_type"),
                                "rerank_score": c.metadata.get("rerank_score", c.metadata.get("similarity")),
                            }
                            for c in all_chunks
                        ],
                        "points": _trace_point_records,
                    })
            except Exception as _trace_exc:
                logger.debug("[ask_stream] contamination trace skipped: %s", _trace_exc)

        # ── Cross-topic contamination via CONVERSATION HISTORY (brief mode) ───
        # The block above only runs in detailed mode and only catches jargon
        # from a RETRIEVED chunk about the wrong topic. A different failure
        # shape hits brief mode: {history} is passed into STRICT_GROUNDED_PROMPT
        # for conversational continuity (pronouns, "the 3rd one"), but the
        # model can pattern-match a structurally similar EARLIER turn in that
        # same history and reuse ITS specific facts instead of this turn's
        # actual full_context. Confirmed live: "can i get the money back if
        # motor insurance matures and my vehicle hasn't met any accident..."
        # (full_context was motor-insurance KB text, no mention of endowment
        # anywhere in it) got answered with "endowment assurance policy...
        # lump sum either on your death" — copied near-verbatim from the
        # PRECEDING turn's life-insurance answer, still visible in history.
        # The KV-cache related-context type-guard elsewhere in this function
        # only blocks contamination arriving via the semantic cache; this
        # catches the same failure shape when the source is THIS session's
        # own history instead. Detection: any specific-type ATTRIBUTION the
        # answer makes (type word directly next to insurance/policy/
        # assurance/cover — see _TYPE_ATTRIBUTION_RE, not the looser
        # _SPECIFIC_TYPE_RE) must appear either in the question/
        # retrieval_query (the user asked about it) or in full_context (the
        # KB actually supports it) — an attribution present in NEITHER has
        # no legitimate source this turn. Bare-word _SPECIFIC_TYPE_RE was
        # tried first and over-fired on ordinary English: a correct takaful-
        # insurance answer saying "risks are shared among a group" (meaning
        # "a group of people," not Group Insurance) got wrongly discarded.
        _history_type_contamination_detected = False
        if not _keyword_detailed:
            try:
                _hist_ans_src = (_corrected_text or _reply_stripped)
                _hist_legit_text = f"{question} {retrieval_query} {_full_context_uncompressed}".lower()
                for _hist_type_m in _TYPE_ATTRIBUTION_RE.finditer(_hist_ans_src.lower()):
                    _hist_type_word = _hist_type_m.group(1).lower()
                    if _hist_type_word not in _hist_legit_text:
                        _hist_refusal_text = (
                            "Hmm, I don't have that specific information in my knowledge base right now. "
                            "Let me get one of our agents on it, they'll be able to help you better! 😊"
                        )
                        _reply_stripped = _hist_refusal_text
                        _kv_reply = _hist_refusal_text
                        _corrected_text = _hist_refusal_text
                        _history_type_contamination_detected = True
                        logger.info(
                            "[ask_stream] discarded brief answer: named %r, absent from "
                            "question/retrieval_query/full_context — likely history bleed",
                            _hist_type_word,
                        )
                        break
            except Exception as _hist_contam_exc:
                logger.debug("[ask_stream] history-contamination filter skipped: %s", _hist_contam_exc)

        # ── Cross-topic contamination via RETRIEVAL (brief mode) ───────────────
        # Distinct from the history-bleed check above, which only fires when
        # the answer's claim is absent from THIS turn's full_context (i.e. it
        # came from a stale prior turn). This catches the case where the
        # wrong-topic content is genuinely IN full_context this turn — a
        # broad "general"-tagged KB chunk spanning multiple insurance types
        # (e.g. a single paragraph listing disclosure examples "(a) In Fire
        # Insurance... (d) In Personal Accident Insurance...") gets retrieved
        # because it matches on shared vocabulary, and the model picks out
        # the wrong sub-item. Confirmed live: "What are the exclusions in
        # term insurance?" answered "...pre-existing conditions, medical
        # history, driving history, and claims history" — "driving history"
        # is a personal-accident/motor underwriting factor from exactly this
        # kind of multi-type chunk, tagged policy_type=general so the
        # metadata retrieval filter doesn't exclude it either. Reuses the
        # same _TYPE_GIVEAWAY_TERMS/_TYPE_QUERY_EXEMPT_WORDS the detailed-mode
        # point filter above uses, applied to the whole reply since brief
        # mode has no discrete points to drop — same discard-and-refuse
        # fallback as the history-bleed check, since a wrong item embedded in
        # a short brief-mode reply usually contaminates most of the value of
        # the answer anyway.
        _retrieval_contamination_detected = False
        if not _keyword_detailed and not _history_type_contamination_detected and _query_policy_type != "general":
            try:
                _retr_ans_src = (_corrected_text or _reply_stripped)
                _retr_query_lower = f"{question} {retrieval_query}".lower()
                if _text_has_giveaway_contamination(
                    _retr_ans_src, _retr_query_lower, _query_policy_type, _full_context_uncompressed
                ):
                    _retr_refusal_text = (
                        "Hmm, I don't have that specific information in my knowledge base right now. "
                        "Let me get one of our agents on it, they'll be able to help you better! 😊"
                    )
                    _reply_stripped = _retr_refusal_text
                    _kv_reply = _retr_refusal_text
                    _corrected_text = _retr_refusal_text
                    _retrieval_contamination_detected = True
                    logger.info(
                        "[ask_stream] discarded brief answer: cross-topic giveaway term found, "
                        "query does not name that type — likely multi-type chunk contamination"
                    )
            except Exception as _retr_contam_exc:
                logger.debug("[ask_stream] retrieval-contamination filter skipped: %s", _retr_contam_exc)

        # ── Strip stray numbered-list markers (brief / conversational mode) ───
        # CONVERSATIONAL_RAG_PROMPT explicitly forbids numbered lists ("No
        # bullet points... Plain conversational prose only"), but the model
        # doesn't reliably follow that — confirmed live repeatedly this
        # session (e.g. "what documents do I need" answers). Worse, whether
        # each marker is separated from the previous point by a newline or
        # just a space is itself inconsistent from one generation to the
        # next, so a genuinely complete answer could render as either a
        # readable list or a confusing run-on ("...manufacture. 3. Copy of
        # your license...") purely by luck — the SAME question asked twice
        # could look fine once and broken the next. Rather than trying to
        # reliably render something the format rules say shouldn't exist in
        # the first place, strip the markers entirely so the text reads as
        # the flowing prose the prompt actually asked for, regardless of
        # which separator the model happened to use.
        # A genuine 2+ point numbered list is now an allowed FORMAT option in
        # brief mode (STRICT_GROUNDED_PROMPT's rule (b) — see prompt_template.py,
        # added so answers with several distinct steps/options don't get
        # crammed into one run-on sentence). This delisting step predates
        # that change and would otherwise silently undo it on every brief
        # answer. Only strip when there are 0-1 numbered markers — a lone
        # stray "1." with no "2." following is never a real list (the
        # original bug this step exists for); 2+ markers is a genuine list
        # and must survive untouched.
        _delist_point_count = len(re.findall(
            r'(?:\A|\n)\s*\d{1,2}\.\s+', (_corrected_text or _reply_stripped)
        ))
        if not _keyword_detailed and _delist_point_count < 2:
            try:
                import re as _delist_re
                _list_src = (_corrected_text or _reply_stripped)
                # A model paragraph break (blank line) right after a list
                # item that has no terminal punctuation of its own signals a
                # sentence boundary the newline-collapse below would
                # otherwise erase — confirmed live: "...Jan Arogya Bima
                # Policy\n\nLet me know if..." and "...Jan Arogya Bima
                # Policy\n\nEach offers different levels..." (an arbitrary
                # model-written wrap-up sentence, not just the fixed
                # sign-off) both lost their separation once collapsed,
                # producing "...Policy Let me know..." / "...Policy Each
                # offers...". Insert the missing period before that happens.
                # Skip when the line already ends in ".!?:" — a colon
                # (introducing the list) or real terminal punctuation needs
                # no extra period. Only run this when the reply actually
                # contains a numbered/bulleted marker — confirmed live this
                # was firing unconditionally on ANY paragraph break in the
                # text, not just ones adjacent to a list. "Sure thing,\n\nHome
                # insurance is a type of..." (the model's own natural lead-in
                # followed by a blank line before its main content, no list
                # anywhere in the reply) got a period wrongly inserted right
                # after the lead-in's comma, producing "Sure thing,. Home
                # insurance..." once newlines collapsed. Gate on the same
                # marker shape _MARKER_RE looks for below so this repair only
                # touches replies that actually need delisting.
                _has_list_marker = bool(
                    _delist_re.search(r'(?:\A|\n)\s*(?:\d{1,2}\.\s+|[-*•]\s+)', _list_src)
                )
                if _has_list_marker:
                    _list_src = _delist_re.sub(r'(?<=[^.!?:\s])\n\s*\n', '.\n\n', _list_src)

                # Join each stripped marker with a space when the preceding
                # text already ends in a sentence boundary (.!?) or a colon
                # (which already introduces the list, so the first item
                # needs no extra separator) — but with ", " otherwise.
                # Confirmed live: "What types of health insurance policies
                # are there?" produced bare noun-phrase items ("1. Mediclaim
                # policy", "2. Overseas Mediclaim policy", ...) with no
                # terminal punctuation of their own. Always joining with a
                # single space (the original behavior) reads fine for full
                # sentences ("...steps in. Pay that bit first...") but glues
                # bare items into an unreadable run-on: "Mediclaim policy
                # Overseas Mediclaim policy Raj Rajeshwari Mahila Kalyan
                # Yojna..." — six policy names with nothing between them.
                # Bullet markers ("- ", "* ", "• ") get the same treatment as
                # numbered markers — confirmed live: "How much does life
                # insurance pay out?" came back as "- **Term Insurance**
                # pays out..." / "- **Whole Life Insurance** pays out..."
                # despite the FORMAT rule banning bullets. Restricted to
                # line starts only (\A or after a newline), unlike numbered
                # markers which also match inline mid-sentence — a bullet
                # dash never legitimately appears inline, but a bare hyphen
                # does ("well-known", "up-to-date"), so matching it inline
                # here would wrongly eat real words.
                _MARKER_RE = _delist_re.compile(
                    r'(?:\A|\n\s*)(?:\d{1,2}\.\s+|[-*•]\s+)'
                    r'|(?<=[.\s])\d{1,2}\.\s+'
                )

                def _join_marker(m):
                    prefix = m.string[:m.start()].rstrip()
                    if not prefix:
                        return ''
                    return ' ' if prefix[-1] in '.!?:' else ', '

                _delisted = _MARKER_RE.sub(_join_marker, _list_src)
                # Any bare newline left after marker-joining is a leftover
                # line break with no list marker attached to it (e.g. the
                # model's own paragraph formatting) — collapse it to a
                # space so the whole reply reads as one flowing paragraph,
                # per FORMAT's "plain conversational prose only".
                _delisted = _delist_re.sub(r'[ \t]*\n[ \t]*', ' ', _delisted)
                _delisted = _delist_re.sub(r'\s{2,}', ' ', _delisted).strip()
                _delisted = _delist_re.sub(r'\s+,', ',', _delisted)
                if _delisted and _delisted != _list_src.strip():
                    _corrected_text = _delisted
                    _kv_reply = _delisted
            except Exception as _delist_exc:
                logger.debug("[ask_stream] numbered-list marker strip skipped: %s", _delist_exc)

        # ── Warm lead-in fallback (brief / conversational mode) ───────────────
        # FORMAT asks for a short warm lead-in ("So,", "Good question,"...)
        # attached to the first sentence, but the model doesn't reliably add
        # one — confirmed live: "What is public liability insurance?" came
        # back as "Public Liability Insurance is a type of coverage..." with
        # no lead-in, reading dry and textbook-like despite everything else
        # (grounding, closing sign-off) being correct. Same lesson as every
        # other formatting-compliance gap in this file: don't trust the
        # prompt instruction alone. Skipped for the small set of known fixed
        # refusal/decline/handoff messages, which already open naturally on
        # their own and explicitly should NOT get an extra lead-in per
        # FORMAT's own carve-out for them.
        if not _keyword_detailed:
            try:
                import re as _lead_re
                _lead_src = (_corrected_text or _reply_stripped).strip()
                _KNOWN_NATURAL_STARTS = (
                    "hmm,", "i'm only set up", "that's a bit outside",
                    "i'm sorry", "no problem!", "sure thing! connecting",
                )
                # Punctuation after the lead-in word varies — "Sure thing!"
                # and "Sure thing," both occur live, not just the comma form
                # every example in this file happens to use. Confirmed live:
                # matching only the comma form let "Sure thing! Fire
                # insurance policies are..." slip past this check, and the
                # fallback below wrongly prepended a second lead-in on top —
                # "So, Sure thing! Fire insurance...". Accept comma,
                # exclamation mark, or period as the separator.
                _LEAD_IN_RE = _lead_re.compile(
                    r'^(so|good question|sure thing|right|ah)[,.!]', _lead_re.IGNORECASE,
                )
                # Ordered by preference; skip a candidate if its own word
                # already appears as a natural mid-sentence transition near
                # the start of the model's own text (confirmed live: forcing
                # "So," in front of a reply whose 2nd sentence already opens
                # with "So, if someone gets injured..." produced an awkward
                # back-to-back "So, ... So, ..." — pick a non-colliding
                # lead-in instead of always defaulting to the first one).
                _LEAD_IN_CANDIDATES = ("So,", "Good question,", "Right,", "Sure thing,")
                if _lead_src:
                    _lead_lower = _lead_src.lower()
                    _has_natural_start = any(_lead_lower.startswith(p) for p in _KNOWN_NATURAL_STARTS)
                    if not _has_natural_start and not _LEAD_IN_RE.match(_lead_src):
                        _check_window = _lead_src[:200].lower()
                        _chosen_lead = _LEAD_IN_CANDIDATES[0]
                        for _cand in _LEAD_IN_CANDIDATES:
                            _cand_word = _cand.rstrip(",").lower()
                            if not _lead_re.search(r'\b' + _lead_re.escape(_cand_word) + r'\b,', _check_window):
                                _chosen_lead = _cand
                                break
                        # The model's own text was written to stand as the
                        # FIRST sentence, so it's capitalized accordingly
                        # ("Totally get why that's confusing."). Prepending a
                        # lead-in without adjusting that leaves an awkward
                        # mid-sentence capital — confirmed live: "So, Totally
                        # get why that's confusing." Lowercase the first
                        # letter unless the first word is the pronoun "I"
                        # (and its contractions) or a genuine multi-letter
                        # acronym (NCB, ULIP, TPA) that should stay as-is.
                        _first_word_m = _lead_re.match(r"^(\S+)", _lead_src)
                        _first_word = _first_word_m.group(1) if _first_word_m else ""
                        _skip_lower = (
                            _first_word == "I" or _first_word.startswith("I'")
                            or (len(_first_word) > 1 and _first_word.isupper())
                        )
                        _lead_src_cased = (
                            _lead_src if _skip_lower or not _lead_src
                            else _lead_src[0].lower() + _lead_src[1:]
                        )
                        _with_lead = f"{_chosen_lead} {_lead_src_cased}"
                        _corrected_text = _with_lead
                        _kv_reply = _with_lead
            except Exception as _lead_exc:
                logger.debug("[ask_stream] warm lead-in fallback skipped: %s", _lead_exc)

        # ── Prose→list conversion for genuinely multi-step answers ────────────
        # STRICT_GROUNDED_PROMPT's FORMAT rule now allows (and prefers) a
        # numbered list for 2+ parallel/sequential items, explicitly naming
        # "First,... if... if... if..." as the exact prose shape to avoid —
        # confirmed live the model still doesn't reliably switch to a literal
        # "1. 2. 3." even with that direct callout ("What if the insurer
        # denies to give me the money...?" kept coming back as "First, review
        # your policy... If you still disagree... If that doesn't work..."
        # across two prompt-wording attempts). Same lesson as every other
        # format-compliance gap in this file: enforce in code, don't trust
        # the instruction alone. Runs after the delisting guard above (so an
        # already-genuine model-written list is left untouched, never
        # double-converted) and after the lead-in fallback (so the first
        # sentence is already a normalized lead-in before this splits on it).
        # Detection: 3+ sentences, excluding the lead-in and sign-off, that
        # each start with a sequential/parallel marker word — a reliable
        # signal of genuinely enumerable content strung into prose rather
        # than one continuous thought. A single incidental "If you have
        # questions..." sentence mixed into otherwise non-enumerable prose
        # won't hit this threshold, so ordinary conditional prose is left
        # alone.
        if not _keyword_detailed and _delist_point_count < 2:
            try:
                import re as _p2l_re
                _p2l_src = (_corrected_text or _reply_stripped).strip()
                _p2l_sentences = [s for s in _p2l_re.split(r'(?<=[.!?])\s+', _p2l_src) if s.strip()]
                if len(_p2l_sentences) >= 4:
                    _p2l_signoff_re = _p2l_re.compile(
                        r"^(hope that|let me know|feel free|hang tight|glad to|dig into any part)",
                        _p2l_re.IGNORECASE,
                    )
                    _p2l_lead_re = _p2l_re.compile(
                        r'^(so|good question|sure thing|right|ah|hmm)[,.!]', _p2l_re.IGNORECASE,
                    )
                    _p2l_marker_re = _p2l_re.compile(
                        r'^(first|second|third|next|then|also|additionally|finally|if)\b',
                        _p2l_re.IGNORECASE,
                    )
                    _p2l_start = 1 if _p2l_lead_re.match(_p2l_sentences[0]) else 0
                    _p2l_end = len(_p2l_sentences)
                    if _p2l_signoff_re.match(_p2l_sentences[-1].strip()):
                        _p2l_end -= 1
                    _p2l_middle = _p2l_sentences[_p2l_start:_p2l_end]
                    _p2l_marker_count = sum(
                        1 for s in _p2l_middle if _p2l_marker_re.match(s.strip())
                    )
                    if _p2l_marker_count >= 3:
                        _p2l_points: list = []
                        _p2l_cur: list = []
                        for _s in _p2l_middle:
                            if _p2l_marker_re.match(_s.strip()) and _p2l_cur:
                                _p2l_points.append(" ".join(_p2l_cur).strip())
                                _p2l_cur = [_s]
                            else:
                                _p2l_cur.append(_s)
                        if _p2l_cur:
                            _p2l_points.append(" ".join(_p2l_cur).strip())
                        if len(_p2l_points) >= 3:
                            _p2l_lead = _p2l_sentences[0] if _p2l_start else ""
                            _p2l_signoff = (
                                _p2l_sentences[-1].strip() if _p2l_end < len(_p2l_sentences) else ""
                            )
                            _p2l_list = "\n".join(
                                f"{i}. {p}" for i, p in enumerate(_p2l_points, 1)
                            )
                            _p2l_pieces = [p for p in (_p2l_lead, _p2l_list, _p2l_signoff) if p]
                            _p2l_rebuilt = "\n\n".join(_p2l_pieces)
                            _corrected_text = _p2l_rebuilt
                            _kv_reply = _p2l_rebuilt
                            logger.info(
                                "[ask_stream] converted %d-marker prose answer into a %d-point numbered list",
                                _p2l_marker_count, len(_p2l_points),
                            )
            except Exception as _p2l_exc:
                logger.debug("[ask_stream] prose-to-list conversion skipped: %s", _p2l_exc)

        # ── Hard sentence cap (brief / conversational mode) ──────────────────
        # Conversational prompts instruct the model to write 3 sentences max.
        # This enforcer guarantees it regardless of model compliance.
        # Skipped when the user explicitly asked for an example: "simple
        # language with example" sets has_simple (not has_detail), so
        # _keyword_detailed is False and this cap used to fire anyway —
        # confirmed live, a simple+example answer (re-explanation + a
        # concrete example, naturally 5-6 sentences) got chopped to 4, and
        # since corrected_text replaces the already-streamed bubble once
        # "done" fires, the user watched the example they asked for appear
        # then vanish. Prioritize not truncating requested content over
        # strict brevity here.
        #
        # Cap raised 4 -> 6: confirmed live again with a plain multi-step
        # "how do I file a claim" question (no example/detail modifier at
        # all) — a genuinely complete, naturally-finished 5-sentence answer
        # (finish_reason='stop') still got chopped to 4, silently dropping a
        # real step. Sequential how-to answers routinely need 4-6 steps to
        # actually be complete; the 300-token budget upstream already bounds
        # runaway length, so this cap only needs to catch actual rambling,
        # not cut a legitimate last step.
        #
        # Cap raised 6 -> 9 -> 14, then made conditional on finish_reason:
        # every one of the 3 raises above was needed because a fixed
        # sentence count can't tell a naturally-completed answer (any
        # length) from actual rambling — a compound two-part question
        # ("what's the difference between fire and burglary insurance, AND
        # which should a small shop owner get") legitimately needed 8, then
        # 11 sentences across two retries, chasing the cap up each time.
        # That's a losing game against generation variance no matter how
        # high the number goes.
        #
        # The real signal was sitting unused in the SSE stream the whole
        # time: OpenAI-compatible completions report finish_reason='stop'
        # on the chunk where the model concluded on its own (as opposed to
        # 'length', hitting the max_tokens ceiling mid-thought). Now
        # captured into _finish_reason as the stream is read (see above).
        # A 'stop' means the model itself decided the answer was complete —
        # trust that regardless of sentence count; this cap's job was
        # always to catch rambling, and a model that stopped on its own
        # isn't rambling. Only a NON-stop completion (hit the token
        # ceiling, or finish_reason wasn't reported at all — the buffered
        # fallback path, or a backend that omits it) still gets the cap,
        # since those are exactly the cases where "many short sentences and
        # no natural conclusion" is a real possibility. Kept at 10 for that
        # narrower population — comfortably covers the documented 5-6-step
        # how-to case without needing another cap-chasing bump later, while
        # still catching truly excessive unstructured output when we have
        # no better signal to go on.
        # Wrapped in try/except so any edge-case failure keeps the original reply.
        if not _keyword_detailed and not _kv_has_example and _finish_reason != 'stop':
            try:
                import re as _re
                _cap_src = (_corrected_text or _reply_stripped).strip()
                if _cap_src:
                    # Simple split: find positions of sentence-ending punctuation
                    # followed by whitespace, then take first N chunks.
                    # (?<!\d\.) excludes a split right after a bare numbered-list
                    # marker ("1.", "2.", "3.") — without it, a genuinely-complete
                    # numbered answer gets miscounted as having way more
                    # "sentences" than it does (each marker counts as its own
                    # fragment on top of its content), so this cap fires when it
                    # shouldn't and cuts mid-list. Confirmed live: a correct,
                    # fully-generated 4-point "what documents do I need" answer
                    # (finish_reason=stop, nothing wrong with the generation)
                    # got miscounted as 10 "sentences" — 4 real ones plus each
                    # "1./2./3./4." marker split off on its own — and the cap
                    # chopped it right after the stray "3." marker, discarding
                    # points 3 and 4 the model had already finished writing.
                    _sent_parts = _re.split(r'(?<=[.!?])(?<!\d\.)\s+', _cap_src)
                    _MAX_SENTENCES = 10
                    if len(_sent_parts) > _MAX_SENTENCES:
                        _capped = " ".join(_sent_parts[:_MAX_SENTENCES]).strip()
                        # Ensure it ends cleanly
                        if _capped and _capped[-1] not in '.!?':
                            _capped += '.'
                        _corrected_text = _capped
                        _kv_reply = _capped
                        logger.info(
                            "[ask_stream] sentence cap trimmed reply (%d -> %d sentences, finish_reason=%r)",
                            len(_sent_parts), _MAX_SENTENCES, _finish_reason,
                        )
            except Exception as _cap_exc:
                logger.debug("[ask_stream] sentence cap skipped: %s", _cap_exc)

        # Set unconditionally (not just inside the try block below) so the
        # hollow-answer detector further down can safely check it even if
        # the currency-filter's try block never runs or exits early.
        _dropped_num = 0

        # ── Drop sentences/points with an ungrounded currency figure ──────────
        # Every prompt's grounding rules say "never state a number... unless
        # that exact figure appears literally in the CONTEXT — no estimates,
        # no exceptions, ever", but the model doesn't reliably follow this
        # when asked for an illustrative example. Confirmed live twice:
        # "explain 'excess' with an example" invented "₹10,000" as a sample
        # amount, and separately "explain Liability Only Policy with an
        # example" invented "$10,000" in medical costs for a pedestrian
        # scenario — neither figure appears anywhere in full_context, and
        # the second one even uses the wrong currency for this KB (this is
        # India-context content that uses ₹, never $). A hypothetical
        # SCENARIO ("imagine you hit a pedestrian...") is exactly what
        # "explain with an example" is asking for and is left untouched;
        # the problem is a SPECIFIC invented figure stated as if real,
        # which a reader could mistake for an actual policy limit or
        # typical claim size rather than an arbitrary placeholder.
        #
        # Scoped to currency figures specifically (₹/$/Rs/INR/EUR + digits) —
        # deliberately NOT percentages or day/year counts, since those
        # routinely appear in this KB written out as words ("within
        # fourteen days") rather than digits, and a naive digit-only check
        # would have no way to confirm a word-form figure is grounded,
        # risking false-positive drops of genuinely correct content.
        # "EUR" added alongside "€" — found live via the qualifier-mismatch
        # investigation below that this KB's Travel Insurance Guide writes
        # amounts as both "€100" and "EUR 100" (2 occurrences of the text
        # form), and the text form wasn't recognized at all before this,
        # silently exempting those figures from every check in this block.
        try:
            import re as _re5
            _CURRENCY_RE = _re5.compile(r'(?:[₹$£€]|\bRs\.?\b|\bINR\b|\bEUR\b)\s?([\d,]+(?:\.\d+)?)', _re5.IGNORECASE)
            _num_src = (_corrected_text or _reply_stripped).strip()
            _ctx_digits = _re5.sub(r'\D', '', _full_context_uncompressed or '')

            def _currency_grounded(num_str: str) -> bool:
                digits = num_str.replace(',', '').split('.')[0]
                return bool(digits) and digits in _ctx_digits

            # A figure passing the digit-presence check above is grounded in
            # the sense that it appears SOMEWHERE in the retrieved text, but
            # says nothing about whether it's the RIGHT figure for what was
            # asked. Confirmed live: "how much would they pay if my luggage
            # gets lost" got answered with "up to EUR 100... when required
            # due to bedbugs" — a real, in-context figure, but scoped to an
            # unrelated narrow clause (luggage CLEANING after a bedbug
            # infestation) sitting in the same KB paragraph as two other
            # distinct compensation clauses (delayed luggage: €80/day up to
            # €320; general lost/damaged property: value-based, no fixed
            # cap) — the model picked the wrong one. Confirmed this isn't a
            # one-off: the KB has at least 3 other currency figures scoped
            # to an explicit narrow "due to X"/"in case of X" condition (a
            # dental-specific $225 cap, an illness-triggered €5,000 cap),
            # each a candidate for the identical misattribution against a
            # differently-phrased question.
            #
            # First attempt at this only checked whether the REPLY's own
            # sentence spelled out a "due to X"/"in case of X" qualifier —
            # too narrow. Confirmed live across repeated retries of the
            # exact same lost-luggage question: the model doesn't reliably
            # repeat the source's own qualifying phrase in its paraphrase.
            # "the insurance will pay up to EUR 100... to cover the cost of
            # necessities" (no "bedbugs" mentioned anywhere in that reply
            # sentence) is just as wrong as the version that DOES say "due
            # to bedbugs" — the reply-text-only check missed it entirely,
            # keeping the fabricated-sounding "necessities" framing paired
            # with the wrong figure.
            #
            # Checks the SOURCE text directly instead of the reply's own
            # phrasing: for each currency figure the reply cites, find that
            # exact figure in full_context, take the 2-sentence window
            # ending at its own sentence (its own sentence plus the one
            # immediately before — this KB's benefit clauses routinely put
            # the trigger condition in the sentence before the amount,
            # e.g. "...if luggage is delayed by more than six hours.
            # Compensation of EUR 80/day is paid... up to EUR 320."), and
            # extract any CLAIM-SCENARIO trigger words found there (lost,
            # delayed, bedbugs, dental, etc. — a closed, project-specific
            # list, not general vocabulary). If that source window names a
            # specific trigger and none of it matches anything in the
            # user's own question, the figure is scoped to a condition the
            # user didn't ask about. A window with NO trigger words at all
            # (a general, unscoped statement) is left alone — this only
            # fires when the source itself signals a narrow condition.
            # Scoped to whichever figure occurrence in context looks most
            # compatible (a number can legitimately appear more than once
            # for different clauses) — only drops when EVERY occurrence of
            # that figure is scoped to something the question didn't ask.
            _TRIGGER_WORDS = frozenset({
                'lost', 'lose', 'losing', 'stolen', 'steal', 'theft', 'damaged',
                'damage', 'delayed', 'delay', 'cancelled', 'cancel', 'cancellation',
                'interrupted', 'interrupt', 'interruption', 'bedbug', 'bedbugs',
                'infest', 'infested', 'missed', 'miss', 'injury', 'injured',
                'illness', 'ill', 'hospitalization', 'hospitalized', 'death',
                'deceased', 'accident', 'breakdown', 'cleaning', 'clean', 'repair',
                'replace', 'replacement', 'evacuation', 'evacuated', 'quarantine',
                'epidemic', 'disaster', 'terrorism', 'conflict', 'snowfall',
                'weather', 'dental', 'crisis',
            })

            def _extract_triggers(text: str) -> set:
                words = _re5.findall(r'\b[a-z]+\b', (text or '').lower())
                found = set()
                for w in words:
                    stem = w[:-1] if w.endswith('s') and len(w) > 4 else w
                    if w in _TRIGGER_WORDS:
                        found.add(w)
                    elif stem in _TRIGGER_WORDS:
                        found.add(stem)
                return found

            def _source_window(ctx: str, start: int, end: int) -> str:
                right = ctx.find('.', end)
                right2 = ctx.find('\n', end)
                _candidates = [r for r in (right, right2) if r != -1]
                right = min(_candidates) if _candidates else len(ctx)
                cur_left = max(ctx.rfind('.', 0, start), ctx.rfind('\n', 0, start))
                prev_left = max(ctx.rfind('.', 0, cur_left), ctx.rfind('\n', 0, cur_left))
                return ctx[prev_left + 1:right + 1]

            _question_triggers = _extract_triggers(question or '')

            def _qualifier_mismatched(unit: str) -> bool:
                if not _question_triggers:
                    return False
                _figs = _CURRENCY_RE.findall(unit)
                if not _figs:
                    return False
                for _fig in _figs:
                    _digits = _fig.replace(',', '').split('.')[0]
                    if not _digits:
                        continue
                    _fig_re = _re5.compile(
                        r'(?:[₹$£€]|\bRs\.?\b|\bINR\b|\bEUR\b)\s?' + _re5.escape(_digits) + r'\b',
                        _re5.IGNORECASE,
                    )
                    _any_occurrence = False
                    _any_compatible = False
                    for _m in _fig_re.finditer(_full_context_uncompressed or ''):
                        _any_occurrence = True
                        _window_triggers = _extract_triggers(_source_window(_full_context_uncompressed, _m.start(), _m.end()))
                        if not _window_triggers or (_window_triggers & _question_triggers):
                            _any_compatible = True
                            break
                    if _any_occurrence and not _any_compatible:
                        return True
                return False

            # Detailed-mode numbered lists are split on the newline before
            # each marker (preserving list structure on rejoin); everything
            # else falls back to the same numbered-list-aware sentence split
            # used by the cap above, rejoined with spaces like normal prose.
            _has_points = bool(_re5.search(r'(?:^|\n)\s*\d+\.\s', _num_src))
            if _has_points:
                _units = _re5.split(r'\n(?=\s*\d+\.\s)', _num_src)
            else:
                _units = _re5.split(r'(?<=[.!?])(?<!\d\.)\s+', _num_src)

            _kept_units, _dropped_num = [], 0
            for _unit in _units:
                _found = _CURRENCY_RE.findall(_unit)
                if _found and (not any(_currency_grounded(f) for f in _found) or _qualifier_mismatched(_unit)):
                    _dropped_num += 1
                    continue
                _kept_units.append(_unit)

            if _dropped_num and _kept_units:
                if _has_points:
                    _point_re5 = _re5.compile(r'^(\s*)(\d+)(\.\s+)(.*)$', _re5.DOTALL)
                    _renumbered5, _next_n = [], 1
                    for _unit in _kept_units:
                        _m = _point_re5.match(_unit)
                        if _m:
                            _renumbered5.append(f"{_m.group(1)}{_next_n}{_m.group(3)}{_m.group(4)}")
                            _next_n += 1
                        else:
                            _renumbered5.append(_unit)
                    _rebuilt5 = "\n".join(_renumbered5).strip()
                else:
                    _rebuilt5 = " ".join(_kept_units).strip()
                    # Checking the literal last character breaks when the
                    # kept text ends in an emoji sign-off ("...details! 😊")
                    # — the emoji, not "!", is _rebuilt5[-1], so this always
                    # added a redundant period after it ("...details! 😊.").
                    # Strip trailing emoji/whitespace first so the check
                    # looks at the actual last word character.
                    _trailing_m = _re5.search(r'([\s\U0001F300-\U0001FAFF☀-➿]+)$', _rebuilt5)
                    _trailing_suffix = _trailing_m.group(1) if _trailing_m else ''
                    _core = _rebuilt5[:len(_rebuilt5) - len(_trailing_suffix)] if _trailing_suffix else _rebuilt5
                    if _core and _core[-1] not in ".!?":
                        _rebuilt5 = _core + "." + _trailing_suffix
                _corrected_text = _rebuilt5
                _kv_reply = _rebuilt5
                logger.info(
                    "[ask_stream] dropped %d unit(s) containing an ungrounded currency figure",
                    _dropped_num,
                )
                # Dropping the FIRST unit often means dropping the sentence
                # that carried the warm lead-in the earlier fallback already
                # added or confirmed was present — confirmed live: dropping
                # a mismatched "So, ... EUR 100 ... due to bedbugs." opener
                # left the reply starting mid-thought with "For other types
                # of loss or damage...", no lead-in at all. Re-run the same
                # check now that the text has changed shape, rather than
                # duplicating the full lead-in-fallback block — this only
                # fires when a drop actually happened, not on every reply.
                if not _keyword_detailed:
                    _rl_src = _corrected_text.strip()
                    _rl_natural_starts = (
                        "hmm,", "i'm only set up", "that's a bit outside",
                        "i'm sorry", "no problem!", "sure thing! connecting",
                    )
                    _rl_lead_re = _re5.compile(
                        r'^(so|good question|sure thing|right|ah)[,.!]', _re5.IGNORECASE,
                    )
                    if _rl_src and not any(_rl_src.lower().startswith(p) for p in _rl_natural_starts) and not _rl_lead_re.match(_rl_src):
                        _rl_candidates = ("So,", "Good question,", "Right,", "Sure thing,")
                        _rl_window = _rl_src[:200].lower()
                        _rl_chosen = _rl_candidates[0]
                        for _rl_cand in _rl_candidates:
                            _rl_word = _rl_cand.rstrip(",").lower()
                            if not _re5.search(r'\b' + _re5.escape(_rl_word) + r'\b,', _rl_window):
                                _rl_chosen = _rl_cand
                                break
                        _corrected_text = f"{_rl_chosen} {_rl_src}"
                        _kv_reply = _corrected_text
        except Exception as _num_exc:
            logger.debug("[ask_stream] ungrounded-currency filter skipped: %s", _num_exc)

        # Set unconditionally for the same reason as _dropped_num above — the
        # hollow-answer detector needs to check this even if the try block
        # below never runs.
        _refund_units_dropped = False

        # ── Drop refund/money-back claims scoped to an unrelated event ────────
        # Confirmed live on both vLLM and Groq (see [[project_groq_4th_attempt
        # _confirms_avoid]]): "will I get my money back if my health policy
        # expires and I never made a claim" got answered with "no refund of
        # premium is allowed if you don't make a claim" — stated as a general
        # rule. The KB source this is drawn from is actually the Group
        # Mediclaim clause "No refund of premium is allowed for deletion of
        # insured person if he or she has recovered a claim under the
        # policy" — a narrow rule about removing a family member from a GROUP
        # policy mid-term, unrelated to a policy simply expiring unused. Both
        # backends generalized a specific, unrelated administrative clause
        # into a blanket answer. Same bug class as the currency-qualifier-
        # mismatch check above (a scoped source clause misapplied to an
        # unstated scenario) but without a currency figure to anchor on, so
        # it needs its own detection: only fires when the user's question is
        # PURELY about natural expiry/non-use (no cancellation/deletion/
        # surrender/free-look wording of its own) and there is no refund
        # statement in the retrieved context that actually applies to that
        # scenario — either because every refund-mentioning sentence found is
        # scoped to one of those unrelated triggering events, OR because the
        # word "refund" never appears in the retrieved context at all (a live
        # retest confirmed this second shape too: with a differently-worded
        # question and a different retrieval, the reply asserted "the insurer
        # doesn't refund premiums if you don't use the coverage" while
        # full_context contained no "refund" text whatsoever — not a
        # misattributed clause, a claim invented from general insurance
        # knowledge with zero textual support). A source sentence that states
        # a refund rule with no scoping qualifier at all is left alone —
        # that's a genuinely general, applicable answer, not a fabrication.
        #
        # Deliberately does not add its own "discard whole reply" branch —
        # dropping units here can legitimately leave nothing but the sign-off
        # ("Let me know if you want more details! 😊") in brief mode, and the
        # hollow-answer detector below already exists specifically to catch
        # that shape and redirect to refusal+escalation (its own comment
        # documents this exact "every other unit dropped upstream by the
        # currency-mismatch filter" interaction).
        try:
            import re as _re6
            _REFUND_CLAIM_RE = _re6.compile(
                r'\brefunds?\b|\bmoney\s*back\b|\bpremium\s+back\b', _re6.IGNORECASE
            )
            _REFUND_SCOPE_TRIGGERS = frozenset({
                'cancellation', 'cancel', 'cancelled', 'canceling', 'cancelling',
                'deletion', 'deleted', 'delete', 'removed', 'removal',
                'free-look', 'freelook', 'free look', 'cooling-off', 'cooling off',
                'surrender', 'surrendered', 'mid-term', 'midterm',
                'dishonour', 'dishonor', 'bounced', 'return the policy',
            })
            _EXPIRY_TRIGGERS = frozenset({
                'expire', 'expires', 'expired', 'expiry', 'lapse', 'lapsed',
                'lapses', 'lapsing', "didn't use", 'did not use', 'not used',
                'unused', 'never used', "didn't claim", 'did not claim',
                "couldn't claim", 'could not claim', 'no claim', 'never claimed',
                'matured', 'maturity',
            })

            def _refund_scope_words(text: str) -> set:
                low = (text or '').lower()
                return {t for t in _REFUND_SCOPE_TRIGGERS if t in low}

            _q_low6 = (question or '').lower()
            _question_is_expiry = any(t in _q_low6 for t in _EXPIRY_TRIGGERS)
            _question_scope = _refund_scope_words(question)

            _refund_scope_mismatch = False
            if _question_is_expiry and not _question_scope:
                _any_compatible_src = False
                for _m in _re6.finditer(
                    r'[^.\n]*\brefund[^.\n]*[.\n]', _full_context_uncompressed or '', _re6.IGNORECASE
                ):
                    if not _refund_scope_words(_m.group(0)):
                        _any_compatible_src = True
                        break
                # Fires whenever no compatible refund statement was found —
                # whether because every occurrence found was scoped away
                # from this scenario, or because "refund" never appears in
                # the retrieved context at all (loop simply never runs).
                _refund_scope_mismatch = not _any_compatible_src

            if _refund_scope_mismatch:
                _rf_src = (_corrected_text or _reply_stripped).strip()
                _rf_has_points = bool(_re6.search(r'(?:^|\n)\s*\d+\.\s', _rf_src))
                if _rf_has_points:
                    _rf_units = _re6.split(r'\n(?=\s*\d+\.\s)', _rf_src)
                else:
                    _rf_units = _re6.split(r'(?<=[.!?])(?<!\d\.)\s+', _rf_src)

                _rf_kept = [u for u in _rf_units if not _REFUND_CLAIM_RE.search(u)]
                if len(_rf_kept) < len(_rf_units) and _rf_kept:
                    if _rf_has_points:
                        _rf_point_re = _re6.compile(r'^(\s*)(\d+)(\.\s+)(.*)$', _re6.DOTALL)
                        _rf_renumbered, _rf_next_n = [], 1
                        for _u in _rf_kept:
                            _m = _rf_point_re.match(_u)
                            if _m:
                                _rf_renumbered.append(f"{_m.group(1)}{_rf_next_n}{_m.group(3)}{_m.group(4)}")
                                _rf_next_n += 1
                            else:
                                _rf_renumbered.append(_u)
                        _rf_rebuilt = "\n".join(_rf_renumbered).strip()
                    else:
                        _rf_rebuilt = " ".join(_rf_kept).strip()
                        if _rf_rebuilt and _rf_rebuilt[-1] not in ".!?":
                            _rf_rebuilt += "."
                    _corrected_text = _rf_rebuilt
                    _kv_reply = _rf_rebuilt
                    _refund_units_dropped = True
                    logger.info(
                        "[ask_stream] dropped refund claim scoped to an unrelated "
                        "triggering event (source conditions the rule on "
                        "cancellation/deletion/surrender/free-look, not plain expiry)"
                    )
        except Exception as _refund_exc:
            logger.debug("[ask_stream] refund-scope-mismatch filter skipped: %s", _refund_exc)

        # Set unconditionally for the same reason as _dropped_num /
        # _refund_units_dropped above.
        _denial_units_dropped = False

        # ── Drop coverage-denial claims not actually named in the source ──────
        # Confirmed live: "if I intentionally damage my car under motor
        # insurance, will I get any money?" got answered "you likely won't
        # get any money... most policies have a clause that excludes
        # intentional damage" — stated as if quoting a specific policy
        # clause. The KB's actual motor-insurance "General Exclusions" list
        # names exactly 5 items (outside the covered geographical area,
        # contractual liability, use outside the policy's Use Clause, no
        # valid driving licence, war/nuclear risks) — "intentional damage"
        # is not one of them. The claim is directionally TRUE (a real,
        # general insurance principle — deliberately causing your own loss
        # voids a claim on moral-hazard/utmost-good-faith grounds) and is
        # even loosely present in the KB as a general "moral hazard" concept
        # passage (an unrelated warehouse-arson example, no cars, no named
        # exclusions list) — but the reply presents it as if it were a
        # specific, citable clause for THIS policy type, which it isn't.
        #
        # Same underlying bug class as the refund-scope-mismatch check
        # above (a real principle applied as if it were specific retrieved
        # text) but deliberately NOT scoped to refunds only — this is
        # general enough to catch any coverage-denial claim ("won't cover
        # X", "excludes X", "not covered") regardless of topic: flood
        # damage, drunk driving, pre-existing conditions, whatever. Rather
        # than enumerate scenario types (the exact whack-a-mole pattern
        # already rejected for modifier-intent detection — see
        # [[project_hybrid_modifier_intent_classifier]]), this drives
        # entirely off the USER'S OWN QUESTION WORDING: strip generic
        # insurance/English words, and check whether any of what's left
        # (the words that actually describe the scenario — "intentionally",
        # "damage") appear anywhere near an exclusion-indicating phrase in
        # the retrieved context. A source window with no scenario word at
        # all is either a general/unconditional exclusion (left alone,
        # matching the currency-check's identical "no qualifier = general,
        # don't touch it" logic) or — if no exclusion-context window
        # anywhere in the document contains the scenario word — the reply's
        # specific claim isn't actually grounded, regardless of how true it
        # sounds.
        try:
            import re as _re7
            # Confirmed live: the first version of this regex only matched
            # "won't"/"will not" immediately followed by the verb — missed
            # "wouldn't" (a different modal, not just a contraction of
            # "won't") entirely, and broke on any filler adverb between the
            # negation and the verb ("wouldn't TYPICALLY cover"). Same
            # fragility class already fixed once this session for identity
            # detection (see [[project_identity_regex_filler_words_gap]]) —
            # permissive of a few filler words between the negation and verb
            # rather than requiring exact adjacency.
            _DENIAL_FILLER = r"(?:\s+(?:typically|usually|generally|normally|likely|probably|necessarily|always|certainly|automatically)){0,2}"
            _DENIAL_RE = _re7.compile(
                r"won'?t" + _DENIAL_FILLER + r"\s+(?:get|cover|pay|receive|include|payout|pay\s+out)|"
                r"will\s+not" + _DENIAL_FILLER + r"\s+(?:get|cover|pay|receive|include|payout|pay\s+out)|"
                r"would\s?n'?t" + _DENIAL_FILLER + r"\s+(?:get|cover|pay|receive|include|payout|pay\s+out)|"
                r"would\s+not" + _DENIAL_FILLER + r"\s+(?:get|cover|pay|receive|include|payout|pay\s+out)|"
                r"does\s?n'?t\s+cover|does\s+not\s+cover|"
                r"\bexcludes?\b|\bexcluded\b|"
                r"not\s+covered|no\s+coverage|not\s+payable",
                _re7.IGNORECASE,
            )
            _EXCLUSION_INDICATOR_RE = _re7.compile(
                r"exclu\w*|not\s+cover\w*|does\s?n'?t\s+cover|will\s+not\s+cover|"
                r"not\s+payable|no\s+coverage",
                _re7.IGNORECASE,
            )
            # Generic insurance vocabulary (already maintained for the
            # hollow-answer detector) plus ordinary English function words —
            # what's left after stripping both is the part of the question
            # that actually names a scenario, not just "insurance" boilerplate.
            _DENIAL_STOPWORDS = frozenset({
                'the', 'a', 'an', 'and', 'or', 'but', 'if', 'will', 'would',
                'can', 'could', 'should', 'is', 'are', 'was', 'were', 'be',
                'been', 'being', 'do', 'does', 'did', 'my', 'your', 'i',
                'you', 'it', 'me', 'to', 'of', 'in', 'on', 'for', 'under',
                'with', 'get', 'any', 'money', 'back', 'that', 'this',
                'what', 'when', 'how', 'why', 'who', 'which', 'have', 'has',
                'not', 'no',
                # Question-structure/meta words — describe what KIND of
                # answer the user wants (an explanation, a list, more
                # detail), not a specific scenario to verify grounding
                # against. Confirmed live: "Explain the exclusions in fire
                # insurance in detail" left {'explain', 'detail',
                # 'exclusions'} as "scenario words" after the existing
                # filters — none of which describe an actual scenario like
                # "intentional damage" this check was built for, and none
                # of which a real KB exclusion clause would ever literally
                # contain. Every genuine exclusion the model correctly
                # listed then failed the "does the source's exclusion
                # window mention this scenario word" test and got dropped,
                # since a general "list the exclusions" question doesn't
                # name one scenario to check against in the first place.
                'explain', 'explains', 'explaining', 'detail', 'details',
                'detailed', 'describe', 'describes', 'list', 'lists',
                'tell', 'give', 'know', 'understand', 'example', 'examples',
                'point', 'points', 'exclusion', 'exclusions', 'exclude',
                'excludes', 'excluded', 'excluding', 'denial', 'denied',
                'deny', 'denies',
            }) | frozenset(w.lower() for term in _INSURANCE_VOCAB for w in term.split())

            def _denial_scenario_words(text: str) -> set:
                words = _re7.findall(r"\b[a-zA-Z]{3,}\b", (text or ''))
                return {w.lower() for w in words if w.lower() not in _DENIAL_STOPWORDS}

            _question_scenario_words = _denial_scenario_words(question)
            # The query's OWN insurance type (e.g. "fire" in "explain the
            # exclusions in fire insurance") isn't a scenario either — it's
            # just naming what the question is about, same non-signal as
            # "explain"/"detail" above. Confirmed live: with those two
            # filtered, "fire" alone survived as the only "scenario word"
            # for a general "explain the exclusions in fire insurance"
            # question, and since formal exclusion clauses don't
            # necessarily repeat the type name in every sentence (a
            # frequency-based pervasive-word filter above didn't
            # reliably catch it either), 3 of 5 genuine exclusions still
            # failed the "does the source window mention this word" check
            # and got dropped.
            if _query_policy_type != "general":
                _question_scenario_words -= set(_query_policy_type.split("_"))
            _dn_reply_src_check = (_corrected_text or _reply_stripped)
            _reply_has_denial = bool(_DENIAL_RE.search(_dn_reply_src_check))

            _denial_claim_mismatch = False
            if _reply_has_denial and _question_scenario_words:
                # A bare policy-type word ("car", "motor") isn't in
                # _INSURANCE_VOCAB but still appears throughout almost every
                # chunk of a topically-retrieved context — confirmed live:
                # "car"/"motor" from the question matched near an UNRELATED
                # exclusion mention purely because those words are pervasive
                # in any motor-insurance passage, not because that specific
                # window discussed the actual scenario asked about. Same
                # "weak discriminator" bug already found once this session
                # in the hollow-answer detector's domain-word check — fixed
                # the same way there (see [[project_hollow_answer_detector]]
                # sibling fix): don't just check presence, check whether the
                # word is actually SCARCE across the full retrieved context.
                # A word appearing 4+ times total is almost certainly a
                # pervasive topic word, not a scenario-specific signal.
                _full_ctx_lower = (_full_context_uncompressed or '').lower()
                _ctx_word_freq = {
                    w: len(_re7.findall(r'\b' + _re7.escape(w) + r'\b', _full_ctx_lower))
                    for w in _question_scenario_words
                }
                _rare_scenario_words = {w for w, c in _ctx_word_freq.items() if 0 < c <= 3}

                _any_compatible_denial_src = False
                _any_exclusion_context = False
                for _m in _EXCLUSION_INDICATOR_RE.finditer(_full_context_uncompressed or ''):
                    _any_exclusion_context = True
                    _window = _full_context_uncompressed[max(0, _m.start() - 300):_m.end() + 300]
                    _window_words = _denial_scenario_words(_window)
                    if not _window_words or (_window_words & _rare_scenario_words):
                        _any_compatible_denial_src = True
                        break
                # Only a real mismatch if the reply is actually making a
                # denial claim AND the document has exclusion-context at all
                # but none of it names this scenario — a document with NO
                # exclusion language anywhere is a different, already-handled
                # case (general ungrounded-claim / hollow-answer territory).
                _denial_claim_mismatch = _any_exclusion_context and not _any_compatible_denial_src

            if _denial_claim_mismatch:
                _dn_src = (_corrected_text or _reply_stripped).strip()
                _dn_has_points = bool(_re7.search(r'(?:^|\n)\s*\d+\.\s', _dn_src))
                if _dn_has_points:
                    _dn_units = _re7.split(r'\n(?=\s*\d+\.\s)', _dn_src)
                else:
                    _dn_units = _re7.split(r'(?<=[.!?])(?<!\d\.)\s+', _dn_src)

                _dn_kept = [u for u in _dn_units if not _DENIAL_RE.search(u)]
                if len(_dn_kept) < len(_dn_units) and _dn_kept:
                    if _dn_has_points:
                        _dn_point_re = _re7.compile(r'^(\s*)(\d+)(\.\s+)(.*)$', _re7.DOTALL)
                        _dn_renumbered, _dn_next_n = [], 1
                        for _u in _dn_kept:
                            _m = _dn_point_re.match(_u)
                            if _m:
                                _dn_renumbered.append(f"{_m.group(1)}{_dn_next_n}{_m.group(3)}{_m.group(4)}")
                                _dn_next_n += 1
                            else:
                                _dn_renumbered.append(_u)
                        _dn_rebuilt = "\n".join(_dn_renumbered).strip()
                    else:
                        _dn_rebuilt = " ".join(_dn_kept).strip()
                        if _dn_rebuilt and _dn_rebuilt[-1] not in ".!?":
                            _dn_rebuilt += "."
                    _corrected_text = _dn_rebuilt
                    _kv_reply = _dn_rebuilt
                    _denial_units_dropped = True
                    logger.info(
                        "[ask_stream] dropped coverage-denial claim not named in "
                        "any exclusion-context window of the retrieved source "
                        "(question scenario words=%r)", _question_scenario_words,
                    )
        except Exception as _denial_exc:
            logger.debug("[ask_stream] unsupported-denial-claim filter skipped: %s", _denial_exc)

        # ── Third-party-victim contamination in first-party examples (brief) ──
        # Confirmed live: "personal accident insurance" — a first-party-only
        # type where the INSURED is always the one who suffers the loss,
        # never the one who harms someone else — got a "give me an example"
        # follow-up answered with a THIRD-PARTY-LIABILITY narrative instead:
        # "imagine you're driving and accidentally hit a pedestrian... the
        # pedestrian would be covered for medical expenses" — describing
        # motor/liability insurance (the policy protects someone the INSURED
        # harmed), the exact opposite of what personal accident insurance
        # does. Reproduced on ~1 in 3 retries of the identical follow-up
        # (nondeterministic, same as every other contamination case in this
        # file). Doesn't name "third party liability" or "motor insurance"
        # literally, so the existing _TYPE_GIVEAWAY_TERMS jargon-match filter
        # above (detailed-mode only, matches type-NAME jargon) wouldn't catch
        # it even if it ran in brief mode — the giveaway here is narrative
        # structure, not vocabulary: the accident victim who gets
        # compensated is someone OTHER than the insured/"you".
        #
        # Scoped to known first-party-only topics (the insured can only ever
        # be the one who suffers the loss) so a genuine third-party-liability
        # or motor-insurance answer, where this narrative is exactly correct,
        # is never touched. A partial "example" that names the wrong
        # beneficiary is actively misleading, not just incomplete, so this
        # redirects straight to the standard refusal+escalation path rather
        # than trying to salvage a fragment — mirrors the existing Rule4-
        # discard/hollow-answer flags so it rides the same, already-proven
        # escalation wiring in the final payload below.
        _tpv_contamination_detected = False
        _FIRST_PARTY_ONLY_RE = _re5.compile(
            r'\bpersonal\s*accident\s*insurance\b|\bhealth\s*insurance\b|'
            r'\blife\s*insurance\b|\bterm\s*insurance\b|\bwhole\s*life\b',
            _re5.IGNORECASE,
        )
        if not _keyword_detailed and _FIRST_PARTY_ONLY_RE.search(retrieval_query or ''):
            try:
                _harm_re = _re5.compile(
                    r'\b(?:hit|hits|hitting|injur\w+|struck?)\s+(?:a\s+)?'
                    r'(pedestrian|another\s+(?:person|driver|vehicle)|third\s*part\w*|someone\s+else)\b',
                    _re5.IGNORECASE,
                )
                _victim_benefit_re = _re5.compile(
                    r'\b(pedestrian|they|the\s+other\s+(?:person|driver|party)|third\s*part\w*)\b'
                    r'[^.!?]{0,80}\b(cover\w*|compensat\w*|reimburs\w*|paid|pay out|indemnif\w*)\b',
                    _re5.IGNORECASE,
                )
                _tpv_src = (_corrected_text or _reply_stripped)
                if _harm_re.search(_tpv_src) and _victim_benefit_re.search(_tpv_src):
                    _refusal_text = (
                        "Hmm, I don't have that specific information in my knowledge base right now. "
                        "Let me get one of our agents on it, they'll be able to help you better! 😊"
                    )
                    _corrected_text = _refusal_text
                    _kv_reply = _refusal_text
                    _tpv_contamination_detected = True
                    logger.info(
                        "[ask_stream] third-party-victim contamination detected in first-party example — redirecting to refusal+escalation"
                    )
            except Exception as _tpv_exc:
                logger.debug("[ask_stream] third-party-victim contamination check skipped: %s", _tpv_exc)

        # ── Hollow-answer detector (brief / conversational mode) ──────────────
        # Confirmed live: a genuinely substantive question ("my claim got
        # rejected and nobody's telling me why, what do I do") retrieved 8
        # sources and passed the grounding check (loose keyword overlap —
        # "claim"/"insurance" appear in plenty of unrelated regulatory
        # boilerplate this KB has about insurer/regulator dispute processes,
        # not a consumer-facing "how do I appeal my own claim" guide), so it
        # never hit the refusal path. But with nothing actually grounded to
        # say, the model correctly followed its "don't invent ungrounded
        # content" rule and produced only empathy: "So, it's really
        # frustrating when that happens. Let me know if you want more
        # details! 😊" — a technically rule-compliant reply that helps the
        # user precisely zero, and (unlike a proper refusal) never sets
        # needs_human, so it doesn't escalate either. The user is left with
        # nothing and no path to a human.
        #
        # Detect this by stripping the lead-in and the one fixed sign-off
        # phrase, then checking what's left: BOTH short (<12 words — well
        # under FORMAT's own 15-25-words-per-sentence target, so this can't
        # false-positive on a legitimately terse-but-complete single
        # sentence) AND missing every word in _INSURANCE_VOCAB (a real
        # answer to an insurance question essentially always uses at least
        # one). Requiring both together, rather than either alone, is what
        # keeps this from misfiring on genuinely short factual answers
        # ("No, travel insurance doesn't cover flight delays." is short but
        # contains "insurance"/"cover"). If both hold, treat it exactly like
        # the existing Rule4-discard refusal case: swap in the standard
        # refusal message and mark needs_human so it escalates properly
        # instead of silently doing nothing.
        _hollow_answer_detected = False
        if not _keyword_detailed:
            try:
                import re as _hollow_re
                _hollow_src = (_corrected_text or _reply_stripped).strip()
                _hollow_leadin_re = _hollow_re.compile(
                    r'^(so|good question|sure thing|right|ah)[,.!]\s*', _hollow_re.IGNORECASE,
                )
                _content_only = _hollow_leadin_re.sub('', _hollow_src, count=1)
                # Was "...want more details!?" only — a single fixed phrase.
                # Confirmed live the model doesn't always use that exact
                # wording ("Let me know if you need any other examples! 😊"
                # survived untouched, inflating the word count enough to
                # dodge this check entirely). The sign-off is a single
                # prompt-mandated closer regardless of its exact phrasing, so
                # strip everything from "let me know if you want/need"
                # onward rather than one literal string.
                _content_only = _hollow_re.sub(
                    r'\s*let me know if you (?:want|need)\b.*$', '', _content_only,
                    flags=_hollow_re.IGNORECASE | _hollow_re.DOTALL,
                ).strip()
                _word_count = len(_hollow_re.findall(r'\w+', _content_only))
                _has_domain_word = any(term in _content_only.lower() for term in _INSURANCE_VOCAB)
                # Gate on the ORIGINAL text being non-empty, not on content
                # surviving the lead-in/sign-off strip — confirmed live: a
                # reply that was NOTHING but lead-in + sign-off ("So, Let me
                # know if you want more details! 😊", every other unit
                # dropped upstream by the currency-mismatch filter) strips
                # down to an empty _content_only, and "empty and < 12 words
                # and no domain word" is the MOST hollow case there is —
                # but gating on `_content_only` being truthy first meant
                # this exact case short-circuited past the check entirely.
                #
                # _has_domain_word alone is too weak a bar once we already
                # know a unit was dropped upstream: confirmed live,
                # "explain excess with an example" had its invented "₹10,000"
                # sentence correctly dropped by the currency-qualifier
                # filter, leaving "So, imagine you have a comprehensive
                # insurance policy for your car." — 10 words, answers
                # nothing, but "insurance"/"comprehensive"/"policy" are all
                # in _INSURANCE_VOCAB, so the AND-gated check let it through.
                # When the currency or refund-scope filter already stripped
                # a unit, don't require the domain-word absence too — a
                # topic-naming stub with nothing else left is exactly the
                # hollow shape those filters are known to produce.
                _prior_unit_dropped = bool(_dropped_num) or _refund_units_dropped or _denial_units_dropped
                if _hollow_src and _word_count < 12 and (not _has_domain_word or _prior_unit_dropped):
                    _refusal_text = (
                        "Hmm, I don't have that specific information in my knowledge base right now. "
                        "Let me get one of our agents on it, they'll be able to help you better! 😊"
                    )
                    _corrected_text = _refusal_text
                    _kv_reply = _refusal_text
                    _hollow_answer_detected = True
                    logger.info(
                        "[ask_stream] hollow answer detected (content=%r, words=%d) — redirecting to refusal+escalation",
                        _content_only, _word_count,
                    )
            except Exception as _hollow_exc:
                logger.debug("[ask_stream] hollow-answer check skipped: %s", _hollow_exc)

        # ── Always-false factual corrections (brief + detailed mode) ──────────
        # Distinct from _TYPE_GIVEAWAY_TERMS (cross-topic leaks): these are
        # claims that are wrong in EVERY context regardless of topic — a
        # coarse grounding YES/NO check doesn't reliably catch a single wrong
        # clause in an otherwise on-topic, well-grounded answer. Kept
        # deliberately small: only add an entry here once confirmed to recur
        # across multiple INDEPENDENT fresh generations of the identical
        # query — most small-model errors are one-off noise (see
        # [[project_vllm_nondeterministic_quality_glitches]]) and don't
        # belong here; this list is for the minority that keep coming back.
        _false_claim_whole_reply_discarded = False
        try:
            _false_claim_src = (_corrected_text or _reply_stripped)
            _false_claim_fixed = _false_claim_src

            # TPA has exactly one correct meaning in this insurance domain —
            # confirmed live: a detailed health-insurance answer said "TPA
            # stands for Third Party Availability" instead of Administrator.
            # Safe unconditional correction, not a prompt instruction (same
            # reasoning as [[project_dont_trust_buried_disclosure_instructions]]:
            # enforce a known fact deterministically in code).
            _TPA_WRONG_EXPANSION_RE = re.compile(
                r"Third[\s-]+Party\s+(?:Availability|Authorization|Agency|Assistance|Association|Assessor)\b",
                re.IGNORECASE,
            )
            _false_claim_fixed = _TPA_WRONG_EXPANSION_RE.sub(
                "Third Party Administrator", _false_claim_fixed
            )

            # Insurers cannot legally cover the insured's own fines, traffic
            # violations, or penalties — a basic insurance-law principle,
            # not a KB-specific fact. Confirmed reproduced in 2 of 3 fresh
            # generations of "explain motor insurance in detail," claiming
            # comprehensive cover "extends to risks such as fines and
            # theft." Drop the whole sentence/point containing the claim
            # (can't safely rewrite arbitrary surrounding phrasing) rather
            # than surgically excise just the word, which would risk
            # leaving grammatical debris ("...risks such as  and theft").
            _FINES_CLAIM_RE = re.compile(r"\bfines?\b", re.IGNORECASE)
            # A line correctly stating fines are NOT covered ("fines and
            # penalties are excluded") is accurate and useful — only drop
            # lines that appear to be CLAIMING coverage, not ones that
            # already deny it. Imperfect for negation far from "fines" in a
            # long sentence, but a meaningful safety margin over a bare
            # word match.
            _FINES_NEGATION_RE = re.compile(
                r"\b(not|n't|except|exclud\w*|never|no\s+cover\w*)\b", re.IGNORECASE
            )

            def _fc_line_wrongly_claims_fines(_ln: str) -> bool:
                return bool(_FINES_CLAIM_RE.search(_ln)) and not _FINES_NEGATION_RE.search(_ln)

            if any(_fc_line_wrongly_claims_fines(_ln) for _ln in _false_claim_fixed.split("\n")):
                _fc_lines = _false_claim_fixed.split("\n")
                _fc_kept_lines = [
                    _ln for _ln in _fc_lines if not _fc_line_wrongly_claims_fines(_ln)
                ]
                if len(_fc_kept_lines) < len(_fc_lines):
                    # Renumber if a middle point of a numbered list was
                    # dropped, so the list doesn't skip a number.
                    _fc_point_re = re.compile(r"^(\s*)(\d{1,2})(\.\s+)(.*)$")
                    _fc_renumbered, _fc_next_n = [], 1
                    for _ln in _fc_kept_lines:
                        _m = _fc_point_re.match(_ln)
                        if _m:
                            _fc_renumbered.append(f"{_m.group(1)}{_fc_next_n}{_m.group(3)}{_m.group(4)}")
                            _fc_next_n += 1
                        else:
                            _fc_renumbered.append(_ln)
                    _false_claim_fixed = "\n".join(_fc_renumbered)
                    logger.info(
                        "[ask_stream] dropped sentence/point claiming fines coverage — "
                        "insurance never covers the insured's own fines/penalties"
                    )
                else:
                    # A single-sentence brief-mode reply with no line breaks
                    # to drop — the whole reply is built around this one
                    # wrong claim, so discard it same as other contamination
                    # checks rather than leave a mangled fragment.
                    _false_claim_fixed = (
                        "Hmm, I don't have that specific information in my knowledge base right now. "
                        "Let me get one of our agents on it, they'll be able to help you better! 😊"
                    )
                    _false_claim_whole_reply_discarded = True
                    logger.info(
                        "[ask_stream] discarded whole reply claiming fines coverage — "
                        "no line boundary to drop just that claim"
                    )

            if _false_claim_fixed != _false_claim_src:
                _corrected_text = _false_claim_fixed
                _kv_reply = _false_claim_fixed
        except Exception as _false_claim_exc:
            logger.debug("[ask_stream] false-claim correction skipped: %s", _false_claim_exc)

        # ── Citation-leak strip: internal document/page labels ─────────────────
        # DETAILED_GROUNDED_PROMPT/STRICT_GROUNDED_PROMPT already tell the
        # model never to repeat the "[Document: filename (Page N)]" labels
        # prepended to context chunks — but a small model doesn't reliably
        # follow that 100% of the time (same class of gap as
        # [[project_dont_trust_buried_disclosure_instructions]]: a prompt
        # instruction alone isn't enough, deterministic code is). Confirmed
        # live: a detailed follow-up answer said "follow the procedures
        # discussed in Document [ea22bdd3a9bf_m4-5f.pdf (Page 17)] to get
        # the claim processed" — a different surface form than the one the
        # prompt rule's own example already covers, showing this leaks in
        # more than one phrasing. Drop the whole unit (point/sentence)
        # containing a leaked filename+page reference rather than trying
        # to surgically edit the citation out and risk a grammatically
        # broken remainder.
        try:
            import re as _re8
            _CITATION_LEAK_RE = _re8.compile(
                r'[\w][\w\-]*\.(?:pdf|docx?|txt|csv|xlsx?)\s*\(?\s*page\s*\d+\s*\)?',
                _re8.IGNORECASE,
            )
            _cl_src = (_corrected_text or _reply_stripped).strip()
            if _CITATION_LEAK_RE.search(_cl_src):
                _cl_has_points = bool(_re8.search(r'(?:^|\n)\s*\d+\.\s', _cl_src))
                if _cl_has_points:
                    _cl_units = _re8.split(r'\n(?=\s*\d+\.\s)', _cl_src)
                else:
                    _cl_units = _re8.split(r'(?<=[.!?])(?<!\d\.)\s+', _cl_src)

                _cl_kept = [u for u in _cl_units if not _CITATION_LEAK_RE.search(u)]
                if len(_cl_kept) < len(_cl_units) and _cl_kept:
                    if _cl_has_points:
                        _cl_point_re = _re8.compile(r'^(\s*)(\d+)(\.\s+)(.*)$', _re8.DOTALL)
                        _cl_renumbered, _cl_next_n = [], 1
                        for _u in _cl_kept:
                            _m = _cl_point_re.match(_u)
                            if _m:
                                _cl_renumbered.append(f"{_m.group(1)}{_cl_next_n}{_m.group(3)}{_m.group(4)}")
                                _cl_next_n += 1
                            else:
                                _cl_renumbered.append(_u)
                        _cl_rebuilt = "\n".join(_cl_renumbered).strip()
                    else:
                        _cl_rebuilt = " ".join(_cl_kept).strip()
                        if _cl_rebuilt and _cl_rebuilt[-1] not in ".!?":
                            _cl_rebuilt += "."
                    _corrected_text = _cl_rebuilt
                    _kv_reply = _cl_rebuilt
                    logger.info(
                        "[ask_stream] dropped %d unit(s) leaking an internal document/page citation",
                        len(_cl_units) - len(_cl_kept),
                    )
        except Exception as _cl_exc:
            logger.debug("[ask_stream] citation-leak strip skipped: %s", _cl_exc)

        # ── Buffered-path single yield (after all corrections) ────────────────
        # The non-streaming (buffered) path above stored the raw answer in
        # _kv_reply but did NOT yield it — we waited until after Rule4 strip,
        # truncation detection, and sentence cap so the client never sees
        # un-vetted text even momentarily.  Yield exactly once, only for the
        # buffered path; the live-streaming path already yielded tokens as
        # they arrived and must NOT hit this yield.
        if not streamed_ok:
            # Buffered path never streams tokens as they're generated, so
            # TTFT here just is total time to this point — still worth
            # recording (None would misleadingly suggest it was never
            # measured) rather than leaving it unset for these requests.
            if _t_ttft_ms is None:
                _t_ttft_ms = round((time.time() - _t_request_start) * 1000)
            yield (_corrected_text or _reply_stripped)

        # ── KV cache write ────────────────────────────────────────────────────
        # Only cache real answers, not handoff/fallback messages.
        _handoff_phrases = (
            "let me get one of our agents",
            "let me get a human agent",
            "i can get one of our agents",
            "i can get a human agent",
            "i don't have that specific info",
            "don't have that specific info",
            "i don't have that info",
            "don't have all the details",
            "don't have that right now",
            "i don't have that right now",
        )
        if _kv_reply and unique_sources and not any(p in _kv_reply.lower() for p in _handoff_phrases) and not _disable_query_cache:
            try:
                _kv.put(
                    _kv_key,
                    {
                        "answer": _kv_reply,
                        "is_general": False,
                        "sources": unique_sources,
                        "detailed": _keyword_detailed,
                        "has_example": _kv_has_example,
                        "has_simple": _kv_has_simple,
                        "top_rerank": max(_top_rerank, 0.0),  # persist decision-time confidence for cache-hit trust
                    },
                    query_embedding=_kv_q_emb,
                    query_text=retrieval_query,
                )
                logger.info("[ask_stream] KV cache stored for query=%r", retrieval_query[:80])
            except Exception as _exc:
                logger.debug("[ask_stream] KV cache write failed: %s", _exc)

        _t_total_ms = round((time.time() - _t_request_start) * 1000)
        _t_postllm_ms = round((time.time() - _t_postllm_start) * 1000)
        # Single consolidated timing line per request rather than one log
        # line per phase — makes it possible to `grep TIMING` and sort/
        # aggregate by whichever phase is actually the bottleneck across a
        # batch of requests, and per-phase numbers that don't sum to the
        # total (there's always some untimed glue code, plus the fallback/
        # standalone-retry tiers upstream aren't separately measured) are
        # expected — "other" below is exactly that gap, not a bug.
        _t_other_ms = _t_total_ms - sum(
            v for v in (_t_retrieval_ms, _t_grounding_ms, _t_llm_ms) if v is not None
        )
        logger.info(
            "[ask_stream] TIMING total=%dms ttft=%s retrieval=%s grounding=%s llm=%s other=%dms "
            "preprocess=%s promptbuild=%s postllm=%s detailed=%s query=%r question=%r",
            _t_total_ms,
            f"{_t_ttft_ms}ms" if _t_ttft_ms is not None else "n/a",
            f"{_t_retrieval_ms}ms" if _t_retrieval_ms is not None else "n/a",
            f"{_t_grounding_ms}ms" if _t_grounding_ms is not None else "n/a",
            f"{_t_llm_ms}ms" if _t_llm_ms is not None else "n/a",
            _t_other_ms,
            f"{_t_preprocess_ms}ms" if _t_preprocess_ms is not None else "n/a",
            f"{_t_promptbuild_ms}ms" if _t_promptbuild_ms is not None else "n/a",
            f"{_t_postllm_ms}ms" if _t_postllm_ms is not None else "n/a",
            _keyword_detailed,
            retrieval_query[:80],
            # The ORIGINAL raw question, not retrieval_query (which gets
            # rewritten for follow-ups, typo correction, query cleaning,
            # etc. — see the multiple "retrieval_query = " reassignments
            # upstream). A latency/regression harness matching log lines
            # back to the request it sent needs a stable, rarely-
            # reassigned handle — question is reassigned only in the
            # narrow compound-question "both" clarify-reply flow, so this
            # is a far more reliable match target than query= above for
            # that purpose specifically.
            question[:80],
        )

        _final_payload: dict = {
            "sources": [] if (_rule4_discarded or _hollow_answer_detected or _tpv_contamination_detected or _history_type_contamination_detected or _retrieval_contamination_detected or _false_claim_whole_reply_discarded) else unique_sources,
            "done": True,
        }
        if _rule4_discarded or _hollow_answer_detected or _tpv_contamination_detected or _history_type_contamination_detected or _retrieval_contamination_detected or _false_claim_whole_reply_discarded:
            _final_payload["needs_human"] = True
        if _corrected_text:
            _final_payload["corrected_text"] = _corrected_text
        yield "\n\n" + _json.dumps(_final_payload)

    # Management methods (keep as before)
    def video_exists(self, url: str) -> bool:
        return self.video_store.url_exists(url)
    def add_video_chunks(self, url: str, chunks: List[Document], title: str = ""):
        self.video_store.add_video_chunks(url, chunks, title=title)
    def delete_video(self, url: str):
        self.video_store.delete_by_url(url)
    def list_videos(self) -> List[str]:
        return self.video_store.list_urls()
    def webpage_exists(self, url: str) -> bool:
        return self.webpage_store.url_exists(url)
    def add_webpage_chunks(self, url: str, chunks: List[Document]):
        self.webpage_store.add_webpage_chunks(url, chunks)
    def delete_webpage(self, url: str):
        self.webpage_store.delete_by_url(url)
    def list_webpages(self) -> List[str]:
        return self.webpage_store.list_urls()