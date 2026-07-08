"""
Unified RAG with strict grounding – no hallucinations.
Supports document filtering with substring matching.
"""
import asyncio
import logging
import os
import re
from typing import List, Tuple, Optional

try:
    from openai import APIConnectionError as _APIConnectionError, APITimeoutError as _APITimeoutError, APIStatusError as _APIStatusError
except ImportError:
    _APIConnectionError = Exception
    _APITimeoutError = Exception
    _APIStatusError = Exception

from rapidfuzz import fuzz, process
from turbovec_store import _rerank_windows

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


def _join_split_compounds(text: str) -> str:
    """Re-join a compound prefix ("re", "co", "un", "under", "over", "non")
    that was typed with a space or hyphen before a word matching an
    _INSURANCE_VOCAB entry — "re insurance" / "re-insurance" -> "reinsurance",
    "under insured" -> "underinsured" — so retrieval sees the same single
    token the knowledge base actually uses, instead of two separate ones.

    Only joins when the word after the prefix fuzzy-matches a vocab entry
    (score >= 80); an unrelated word right after "re"/"co"/etc. ("re apply",
    "co pay" -> "pay" is too short to check) is left untouched.
    """
    def _try_join(m: re.Match) -> str:
        prefix, rest = m.group(1), m.group(2)
        result = process.extractOne(rest, _INSURANCE_VOCAB, scorer=fuzz.ratio)
        if result is not None:
            best_match, score, _ = result
            if score >= 80:
                return f"{prefix.lower()}{best_match}"
        return m.group(0)

    return _COMPOUND_PREFIX_RE.sub(_try_join, text)


def _correct_typos(text: str) -> str:
    """Fix common typos in insurance-domain terms using fuzzy matching.

    Splits *text* into words. For each word of length >= 4, finds the best
    match in ``_INSURANCE_VOCAB`` using ``rapidfuzz`` with ``fuzz.ratio``.
    If the best match score >= 80 and is not already exact, replaces the
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
    """
    text = _join_split_compounds(text)
    words = text.split()
    corrected = []
    for w in words:
        stripped = w.strip(".,!?;:()[]{}'\"")
        if len(stripped) >= 4:
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
                if (score >= 80 and best_match != stripped
                        and not is_compound_extension and not is_truncated_vocab_word):
                    prefix = w[:len(w) - len(w.lstrip(".,!?;:()[]{}'\""))]
                    suffix = w[len(w.rstrip(".,!?;:()[]{}'\"") or len(w)):]
                    corrected.append(f"{prefix}{best_match}{suffix}")
                    continue
        corrected.append(w)
    return " ".join(corrected)


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
        top_rerank = max(
            (d.metadata.get("rerank_score", 0) for d in docs if hasattr(d, "metadata")),
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


def _needs_detailed_answer(question: str) -> bool:
    """True when the question expects a comprehensive, multi-part, or procedural answer."""
    q = question.lower()
    if any(sig in q for sig in _SIMPLE_SIGNALS):
        return False  # user explicitly wants brief — short-circuit before detail check
    if any(sig in q for sig in _DETAIL_SIGNALS):
        return True
    # Long questions (>25 words) almost always need more than 4 sentences
    if len(question.split()) > 25:
        return True
    # Three or more sub-questions joined by "and"
    if q.count(' and ') >= 3:
        return True
    return False


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


_CHIP_STOPWORDS = frozenset({
    "what", "is", "are", "the", "a", "an", "how", "why", "when", "where",
    "which", "does", "do", "can", "will", "would", "should", "be", "to",
    "of", "in", "for", "and", "or", "on", "at", "by", "with", "that",
    "this", "it", "its", "has", "have", "had", "was", "were", "i", "me",
    "my", "your", "their", "there", "from", "about", "if", "tell", "explain",
    "give", "get", "any", "more",
})


def _question_answerable_in_context(question: str, context: str) -> bool:
    """
    Return True when at least half the content words in *question* appear
    somewhere in *context*.  Cheap, no LLM call.  Content words = words
    longer than 3 chars that are not stopwords.
    """
    words = [w.lower().strip("?.,!;:'\"()") for w in question.split()]
    content_words = [w for w in words if len(w) >= 4 and w not in _CHIP_STOPWORDS]
    if not content_words:
        return False            # can't verify — drop it to be safe
    ctx_lower = context.lower()
    hits = sum(1 for w in content_words if w in ctx_lower)
    return hits >= max(1, len(content_words) * 0.5)


async def _backend_completion(
    prompt: str, max_tokens: int, timeout: float, temperature: float = 0,
    backend_override: Optional[str] = None,
) -> Optional[str]:
    """Fast, non-streaming chat completion against whichever backend is
    currently active (vLLM or Groq — both OpenAI-compatible, same request
    shape). Returns the raw response text, or None on any failure, timeout,
    or unconfigured/unsupported backend.

    Used for short, best-effort auxiliary calls (topic extraction, query
    reformulation, suggested-question generation) that should follow the
    SAME backend as the main answer generation, not be pinned to vLLM
    regardless of FORCE_BACKEND. An earlier version hardcoded these to
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


async def _generate_suggestions(question: str, answer: str, context: str = "") -> list:
    """
    Generate follow-up chip questions grounded in the answer just given.

    Steps:
      1. Ask the LLM to produce 5 candidate questions from the answer text.
      2. Cheap pre-filter: drop candidates whose content words don't even
         lexically appear in the retrieved context (no LLM call).
      3. Semantic verification (_verify_suggestions_grounded): a candidate
         can pass step 2 on a passing mention alone — e.g. the answer says
         "...follows Islamic principles and Sharia law..." in one sentence,
         which is enough word-overlap for "What are Islamic principles?" to
         look "covered", but the KB has no dedicated content actually
         explaining Islamic principles, only insurance content that
         mentions them in passing. Confirmed live: this exact suggestion
         was shown, and clicking it produced a refusal — the KB genuinely
         can't answer it. Step 2 alone can't tell "mentioned in passing"
         from "actually explained", so a real judgment call (the same kind
         _verify_grounding already makes for real questions) is needed
         before a suggestion reaches the user.
      4. Return up to 3 verified questions.
    """
    try:
        if not context or not context.strip():
            return []
        if not answer or not answer.strip():
            return []
        # Ground generation in the ANSWER the user actually read, not the raw
        # retrieved-chunk pool. context[:1000] used to be fed to the LLM
        # instead — but retrieval returns several chunks (often 2000-4400
        # chars each), so the first 1000 chars is effectively just whichever
        # chunk landed first in the pool, which is frequently NOT what the
        # conversational answer was actually grounded in (e.g. a thin-KB
        # topic like Takaful pulling in an unrelated "types of insurance"
        # chunk that also mentions motor insurance as an example). The
        # `answer` param was already being passed in but never referenced in
        # the prompt, so suggestions had no connection to what the user
        # actually read. Context is still used below, only for verification.
        ans_snippet = answer[:800].strip()
        prompt = (
            f"User asked: {question}\n\n"
            f"They were given this answer:\n{ans_snippet}\n\n"
            "Write 5 short follow-up questions (max 8 words each) that:\n"
            "1. Ask about specific facts, terms, or details actually mentioned in the answer above — not other topics.\n"
            "2. A user would naturally ask AFTER reading that exact answer.\n"
            "3. Do not repeat the original question or invent unrelated topics.\n"
            "4. Each question must name the specific product/topic (e.g. the insurance type) instead of using 'it'/'its' — a reader must understand it with no other context, since clicking it sends it as a brand-new question.\n"
            "Output only the questions, one per line, no numbering, no bullets, no explanations:"
        )
        _text = await _backend_completion(prompt, max_tokens=120, timeout=12)
        if not _text:
            return []
        # Parse and clean candidates
        _candidates = [
            q.strip().lstrip("0123456789.-) ").strip()
            for q in _text.split("\n") if q.strip()
        ]
        _candidates = [q for q in _candidates if 2 <= len(q.split()) <= 10]
        # Cheap lexical pre-filter before spending an LLM call on verification
        _candidates = [q for q in _candidates if _question_answerable_in_context(q, context)]
        if not _candidates:
            return []
        _verified = await _verify_suggestions_grounded(_candidates, context)
        return _verified[:3]
    except Exception:
        pass
    return []


async def _verify_suggestions_grounded(candidates: List[str], context: str) -> List[str]:
    """Semantic filter for suggested follow-up chips — same judgment
    _verify_grounding() makes for a real question ("can this be FULLY
    answered from this context, not just topically adjacent"), batched into
    one call for all candidates instead of one round-trip per candidate so
    it doesn't multiply suggestion-generation latency by the candidate
    count. Fails safe to an empty list on any error, timeout, or
    unparseable response — showing no chips beats showing ones that dead-end.
    """
    if not candidates:
        return []
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(candidates))
    # _GROUNDING_CONTEXT_CHARS (3000) is tuned for _verify_grounding()'s single-
    # question check and is only half of the ~6000-char budget the context
    # compressor actually gives full_context here. Truncating to 3000 made this
    # verifier blind to whichever half of the real context didn't happen to
    # come first — confirmed live: it rejected "What does cargo insurance
    # cover?" and "What is hull insurance for?" as ungrounded, yet asking them
    # directly answered both correctly from dedicated KB content the verifier
    # never saw. Use the full budget so it judges against everything the
    # answer was actually grounded in.
    prompt = (
        f"Context:\n{context[:_SUGGESTION_VERIFY_CONTEXT_CHARS]}\n\n"
        f"Candidate follow-up questions:\n{numbered}\n\n"
        "Judge each numbered question SEPARATELY and independently — one "
        "question being unanswerable must NOT affect your judgment of the "
        "others. For each one, could it be FULLY and SPECIFICALLY answered "
        "using ONLY the context above — not just a topic the context "
        "mentions in passing, but enough detail there to actually answer "
        "it? Respond with ONLY the numbers of the questions that CAN be "
        "fully answered, comma-separated (e.g. \"2,3\"). If none can, "
        "respond with exactly: none"
    )
    try:
        raw = await _backend_completion(prompt, max_tokens=20, timeout=12)
    except Exception:
        return []
    if not raw or raw.strip().lower() == "none":
        return []
    keep_indices = {
        int(tok) - 1 for tok in re.findall(r"\d+", raw)
        if tok.isdigit() and 0 < int(tok) <= len(candidates)
    }
    return [q for i, q in enumerate(candidates) if i in keep_indices]


def _is_likely_followup(question: str) -> bool:
    """Heuristic: is this question likely a follow-up referencing a previous topic?"""
    words = question.strip().split()
    if len(words) > 12:
        return False  # long questions are usually self-contained
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
    if any(sig in q_lower for sig in _all_modifier_signals) or _wants_example(q_lower):
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
        if _GENERIC_PROCESS_RE.search(q_lower) and not _SPECIFIC_TYPE_RE.search(q_lower):
            return True

    return False


# Specific insurance-type nouns, used only to deterministically re-anchor a
# reformulated follow-up whose LLM rewrite silently dropped the topic — see
# the repair step inside _reformulate_query for why this exists.
_ANCHOR_TYPE_RE = re.compile(
    r"\b(term|motor|car|vehicle|auto|bike|two.wheeler|four.wheeler|"
    r"life|whole\s*life|endowment|ulip|"
    r"health|medical|"
    r"travel|trip|"
    r"home|house|property|"
    r"marine|cargo|fire|"
    r"liability|third.party|"
    r"critical\s*illness|"
    r"group|corporate|"
    r"personal\s*accident|disability|"
    r"retirement|pension|annuity|takaful|"
    r"crop|agricultur\w*)\b(\s+insurance)?",
    re.IGNORECASE,
)


def _last_anchor_type_match(text: str) -> Optional[str]:
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
        user_matches = list(_ANCHOR_TYPE_RE.finditer(user_turns[-1]))
        if user_matches:
            phrase = user_matches[-1].group(0).strip()
            if not re.search(r"insurance\s*$", phrase, re.IGNORECASE):
                phrase = f"{phrase} insurance"
            return phrase

    matches = list(_ANCHOR_TYPE_RE.finditer(text))
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
_REGULATORY_BOILERPLATE_RE = re.compile(
    r"\b(the act|the bill|stamp duty|income tax act|central government|"
    r"section\s+\d+[a-z]?\b.{0,20}\bact\b|"
    r"guarantee[sd]?\s+by\s+the\s+(central\s+)?government|"
    r"amends?\b|gross\s+total\s+income|"
    r"icp\s*\d+|the\s+supervisor)\b",
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
        if _REGULATORY_BOILERPLATE_RE.search(content):
            return 1
        if topic and topic not in content:
            return 1
        return 0
    return sorted(chunks, key=_rank)


async def _reformulate_query(question: str, history: str) -> str:
    """Rewrite a follow-up question as a standalone, natural-language question
    using conversation history — used both for retrieval and, for detected
    follow-ups, as the literal question text shown to the generation prompt.

    Examples:
      history: "User: tell me about life insurance\nLayla: Life insurance ..."
      question: "what about premiums?"
      returns: "What is the premium amount for life insurance?"

    Uses max_tokens=30 and a 4-second timeout so it adds <0.5 s to latency.
    Falls back to the original question on any error.
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
            for _ in range(3):  # rescan after each strip; bounded
                _invented = next(
                    (
                        _m for _m in _ANCHOR_TYPE_RE.finditer(reformulated)
                        if _m.group(2)
                        and not re.search(
                            rf"\b{re.escape(_m.group(1))}\b", _known_topics, re.IGNORECASE
                        )
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
            _hist_anchor = _last_anchor_type_match(recent)
            if (
                _hist_anchor
                and not _ANCHOR_TYPE_RE.search(question)
                and not _ANCHOR_TYPE_RE.search(reformulated)
            ):
                logger.info(
                    "[REFORM] topic-anchor repair: %r missing %r from history — appending",
                    reformulated, _hist_anchor,
                )
                reformulated = f"{reformulated.rstrip('?.! ')} for {_hist_anchor}?"
            logger.info("[REFORM] %r -> %r", question, reformulated)
            return reformulated
    return question


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


# Context passed to the grounding-check prompt is capped so this stays a
# fast, cheap call (max_tokens=10 output either way) — not specified by the
# original design, but consistent with how _generate_suggestions() and the
# earlier context[:1000] pattern elsewhere in this file bound auxiliary-call
# input size for latency.
_GROUNDING_CONTEXT_CHARS = 3000


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
        parts.append(windows[1] if len(windows) > 1 else windows[0])
    return "\n\n".join(parts)

# _verify_suggestions_grounded() judges candidates against the SAME full_context
# the answer was generated from (already capped at ~6000 chars by the context
# compressor's budget), not a fresh single-question lookup — so it needs the
# whole thing, not the smaller 3000-char slice tuned for _verify_grounding().
_SUGGESTION_VERIFY_CONTEXT_CHARS = 6000


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
        "NO only if the context is about a different, unrelated topic, or "
        "the question asks for a specific fact (a country, provider, "
        "number) that is simply absent. "
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
    "I don't have that in my knowledge base right now — "
    "let me get a human agent to help you! 😊"
)


def _strip_markdown(text: str) -> str:
    """Convert markdown-formatted LLM output to plain conversational prose.

    The chat prompt forbids bullet points and bold, but the model sometimes
    ignores that — especially when the retrieved context itself contains
    formatted content. This cleanup runs on every token so the user never
    sees raw markdown.
    """
    import re
    # Remove bold/italic markers
    text = re.sub(r'\*{1,3}([^*]+)\*{1,3}', r'\1', text)
    # Remove ATX headers (## Heading -> Heading)
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    # Convert bullet list items to flowing prose: "- item" or "* item" -> "item, "
    text = re.sub(r'\n\s*[-*]\s+', ' ', text)
    # Numbered list items: "\n1. " -> " "
    text = re.sub(r'\n\s*\d+\.\s+', ' ', text)
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

# Matches the filler-word usage of "honest," (comma right after, functioning
# as a sentence-starting interjection) — not a legitimate adjective use like
# "an honest answer". See the ask_stream call site for why this is fixed
# deterministically instead of relying on the prompt instruction alone.
_HONEST_FILLER_RE = re.compile(r"\b([Hh])onest,")


def _strip_rule4_fallback(text: str, trust_content: bool = True) -> Optional[str]:
    """Handle the LLM appending the canned Rule 4 fallback ("Honestly/Honest,
    I don't have that specific info...") after already writing something, or
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

class MultiSourceRAG:
    def __init__(self):
        self.doc_pipeline = RAGPipeline()
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
                    boost = await asyncio.to_thread(
                        self.doc_pipeline._vector_store.search,
                        retrieval_query, 2, {"source": {"$eq": src}}, True, False,
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
                "I'm Layla, your insurance assistant! I can only help with insurance-related questions — things like policy coverage, premiums, claims, and benefits. Is there something about your insurance I can help you with today? 😊",
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
                answer = "I'm sorry, I can't process that calculation right now — the AI model server seems to be unreachable. Please try again in a moment!"
            return _strip_markdown(_strip_model_preamble(answer)), list(dict.fromkeys(sources)), needs_human, is_off_topic

        if not full_context.strip():
            # No documents retrieved — return a firm refusal rather than letting
            # the LLM answer from training knowledge (small 7B models ignore grounding
            # instructions when context is empty).
            return (
                "Hmm, I don't have that specific information in my knowledge base right now. "
                "Let me get one of our agents on it — they'll be able to help you better! 😊",
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
                "Let me get one of our agents on it — they'll be able to help you better! 😊",
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
                    "pull from general knowledge either. Try again in a moment — or feel free to "
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
                yield "Sure thing! Connecting you with a human agent now — one moment. 😊"
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
        _pasted_grounds_answer = False
        if pasted_context:
            _pasted_grounds_answer = await _verify_grounding(question, pasted_context)

        # ── Typo correction for retrieval query ───────────────────────────────
        # Apply _correct_typos() to build a corrected question used ONLY for
        # the retrieval_query, not for the final LLM prompt (original question
        # is preserved).  This ensures typos like "deductable" → "deductible"
        # before vector search.
        corrected_question = _correct_typos(question)

        # ── Follow-up reformulation ───────────────────────────────────────────
        # For short/vague follow-ups ("what about premiums?", "give me in detail",
        # "is it legal?") rewrite into a full standalone query using history.
        _detected_as_followup = bool(history and _is_likely_followup(question))
        _is_followup = False
        if _detected_as_followup:
            retrieval_query = await _reformulate_query(question, history)
            if retrieval_query != question:
                _is_followup = True
            else:
                # LLM reformulation failed (timeout / vLLM busy) — fall back to
                # merging the follow-up with the last assistant turn, via the
                # same helper ask() already uses for this. NOT just "the last
                # user question" (an earlier version of this fallback): that
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
                _merged = _reformulate_with_history(question, history)
                if _merged and _merged.lower() != question.lower():
                    retrieval_query = _merged
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
        _keyword_detailed = _needs_detailed_answer(question)
        _doc_top_k   = 14 if _keyword_detailed else 8
        _chunk_limit = 12 if _keyword_detailed else 8
        # Trimmed from 5/4 — video/webpage search()'s internal reranking
        # candidate pool is 2x this value (safe_k = min(2*top_k, count)), so
        # this alone controls how many chunks each source reranks before
        # picking the best ones. Fewer candidates = faster reranking, traded
        # against a small chance of missing a good chunk further down the
        # similarity ranking.
        _media_top_k = 4 if _keyword_detailed else 3

        # Apply typo correction to the retrieval_query before KV cache and retrieval
        retrieval_query = _correct_typos(retrieval_query)

        # ── KV cache lookup ───────────────────────────────────────────────────
        # Key includes reformulated query + intent flags so "why is X compulsory"
        # and "why is X compulsory, explain with example" never share a cache entry —
        # even though their retrieval_query is identical (example is a prompt modifier,
        # not a retrieval term, so it doesn't survive reformulation).
        _kv = self.doc_pipeline._cache
        _kv_sources = self.doc_pipeline._vector_store.list_sources()
        _q_lower_intent = question.lower()
        _kv_has_example = _wants_example(_q_lower_intent)
        _kv_has_simple  = any(sig in _q_lower_intent for sig in _SIMPLE_SIGNALS)
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
        _kv_hit = _kv.get(_kv_key)
        if _kv_hit is None:
            try:
                _kv_q_emb = await asyncio.to_thread(
                    lambda: self.doc_pipeline._vector_store.embed_model.encode(
                        [retrieval_query], normalize_embeddings=True, show_progress_bar=False
                    )[0]
                )
                # Higher threshold for detailed queries so "life insurance in detail"
                # doesn't hit "health insurance in detail" just because topics are close.
                _sem_thr = 0.90 if _keyword_detailed else None
                _sem_thr_actual = _sem_thr if _sem_thr is not None else 0.92

                # ── Semantic exact hit: same intent → serve directly ──────────
                _kv_hit = _kv.semantic_get(_kv_q_emb, threshold=_sem_thr)
                if _kv_hit is not None and _kv_hit.get("detailed") != _keyword_detailed:
                    _kv_hit = None
                if _kv_hit is not None and _kv_hit.get("has_example") != _kv_has_example:
                    _kv_hit = None
                if _kv_hit is not None and _kv_hit.get("has_simple") != _kv_has_simple:
                    _kv_hit = None

                # ── Semantic related: different question, overlapping topic ───
                # Collect entries in [0.60, sem_threshold) — related but not
                # identical.  Feed them to the LLM as supplementary context
                # alongside fresh KB chunks; never short-circuit the answer.
                if _kv_hit is None:
                    _related = _kv.semantic_get_related(
                        _kv_q_emb,
                        lower_threshold=0.60,
                        upper_threshold=_sem_thr_actual,
                        top_k=2,
                    )
                    _rel_parts = []
                    _no_ans_phrases = (
                        "i don't have that", "don't have that specific",
                        "let me get one of our agents", "i can get one of our agents",
                        "i can get a human agent", "let me get a human agent",
                    )
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
                        if _q_txt and _a_txt and not any(p in _a_txt.lower() for p in _no_ans_phrases):
                            _rel_parts.append(f"Q: {_q_txt}\nA: {_a_txt}")
                    if _rel_parts:
                        _kv_related_ctx = "\n\n".join(_rel_parts)
            except Exception:
                _kv_q_emb = None
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
            _cached_sugg = _kv_hit.get("suggested_questions")
            if _cached_sugg:
                _cache_payload["suggested_questions"] = _cached_sugg
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
        if not document_filter:
            # Local reranking (doc, video, webpage) all share one CrossEncoder
            # model running on CPU. Firing all three concurrently via
            # asyncio.gather looks parallel but isn't a clean speedup here —
            # each call's own internal PyTorch/OpenMP threading tries to use
            # every available core, so "concurrent" execution causes
            # contention (isolated measurement on this deployment: ~16.6s
            # concurrent vs ~9.8s sequential for the same reranking work).
            # Tried capping each call's thread budget via OMP_NUM_THREADS
            # instead of serializing — that caused requests to hang outright
            # (reranker calls stuck at 0% progress, a real deadlock under
            # concurrent low-thread-count execution), so that approach was
            # reverted. Serializing avoids the deadlock risk entirely and is
            # never worse than the contended-concurrent version, though
            # end-to-end wins are inconsistent in practice — actual per-query
            # cost varies a lot because the stage-1 summary-boost step inside
            # _retrieve_doc_chunks() pulls in a variable, sometimes large,
            # number of extra candidates before reranking (measured 8 vs an
            # actual 18 for the same query), so this alone does not reliably
            # hit any specific latency target — it removes one real risk
            # (contention/deadlock), not the whole bottleneck.
            #
            # The LLM topic-extraction call has no such conflict — it's a
            # network call, not CPU-bound — so it still runs in genuine
            # parallel via its own task while the CPU-bound retrieval work
            # below proceeds sequentially.
            _topics_task = asyncio.create_task(_extract_intent_topics(question))
            doc_chunks = await self._retrieve_doc_chunks(retrieval_query, filter_meta, document_filter, doc_top_k=_doc_top_k, summary_top_k=2)
            video_chunks = await asyncio.to_thread(self.video_store.search, retrieval_query, top_k=_media_top_k, use_hybrid=True, use_reranker=True)
            webpage_chunks = await asyncio.to_thread(self.webpage_store.search, retrieval_query, top_k=_media_top_k, use_hybrid=True, use_reranker=True)
            llm_topics = await _topics_task
            all_chunks = self._merge_chunks(doc_chunks + video_chunks + webpage_chunks)
        else:
            doc_chunks, llm_topics = await asyncio.gather(
                self._retrieve_doc_chunks(retrieval_query, filter_meta, document_filter, doc_top_k=_doc_top_k, summary_top_k=2),
                _extract_intent_topics(question),
            )
            all_chunks = self._merge_chunks(doc_chunks)
        detailed = _keyword_detailed

        # Prefer BGE rerank_score when available (set by rerank_documents).
        # Fall back to the raw retrieval similarity so non-reranked sources
        # (video, webpage) are still ordered reasonably.
        # Stage-1 boost chunks break ties upward: when two chunks share the
        # same rerank_score, the boost chunk (added because its document was
        # explicitly matched by the summary search) should win.
        all_chunks.sort(
            key=lambda x: (
                x.metadata.get("rerank_score", x.metadata.get("similarity", 0)),
                1 if x.metadata.get("stage1_boost") else 0,
            ),
            reverse=True,
        )
        all_chunks = all_chunks[:_chunk_limit]

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
        if not document_filter and _top_rerank < _rerank_gate and not _pasted_grounds_answer:
            logger.info(
                "[ask_stream] Reranker gate: top=%.3f < gate=%.3f — not in KB",
                _top_rerank, _rerank_gate,
            )
            yield (
                "Hmm, I don't have that specific information in my knowledge base right now. "
                "Let me get one of our agents on it — they'll be able to help you better! 😊"
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
        _lex_ok, _semantically_grounded = await asyncio.gather(
            _lexical_covered(),
            _verify_grounding_any_chunk(retrieval_query, all_chunks),
        )
        ctx_covered = _lex_ok and _semantically_grounded

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
                        all_chunks = _fallback_chunks
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
        _question_tokens = {w.lower().strip('?.,!') for w in question.split()}
        _has_unresolved_pronoun = bool(_question_tokens & _FOLLOWUP_SIGNALS)
        _has_unresolved_point_ref = _extract_point_number(question) is not None
        if (
            not ctx_covered and not document_filter and not _pasted_grounds_answer
            and _detected_as_followup and not _has_unresolved_pronoun and not _has_unresolved_point_ref
        ):
            _standalone_chunks = await self._retrieve_doc_chunks(
                question, filter_meta, document_filter, doc_top_k=_doc_top_k, summary_top_k=2,
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
                    all_chunks = _standalone_chunks
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
            yield (
                "Hmm, I don't have that specific information in my knowledge base right now. "
                "Let me get one of our agents on it — they'll be able to help you better! 😊"
            )
            yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": True})
            return

        # Dynamic context budget — scale back when history is long so the total
        # prompt (template + history + context + answer) stays within the model's
        # context window (~4096 tokens for Qwen2.5-7B; use 3900 as safe ceiling).
        # 4 chars ≈ 1 token (Qwen SentencePiece approximation).
        _MAX_INPUT_TOKENS = 3900
        _PROMPT_TEMPLATE_TOKENS = 700       # boilerplate across all prompt templates
        _output_reserve = 1500 if detailed else 300
        _history_tokens_est = len(history) // 4 if history else 0
        _context_token_budget = max(
            300,  # always keep at least a bit of context
            _MAX_INPUT_TOKENS - _PROMPT_TEMPLATE_TOKENS - _history_tokens_est - _output_reserve,
        )
        _context_budget = min(6000, _context_token_budget * 4)  # tokens → chars
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
            context_parts.append(f"[{label}]\n{chunk.page_content}")

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
        if pasted_context:
            if _pasted_grounds_answer:
                full_context = "[Context from an earlier answer]\n" + pasted_context
                # The answer draws only from this, not from retrieval —
                # citing KB documents that were fetched but never actually
                # used would be a misleading source list.
                sources = []
            else:
                full_context = (
                    "[Context from an earlier answer]\n"
                    + pasted_context
                    + ("\n\n[Knowledge base chunks]\n" + full_context if full_context.strip() else "")
                )

        if not full_context.strip():
            yield (
                "Hmm, I don't have that specific information in my knowledge base right now. "
                "Let me get one of our agents on it — they'll be able to help you better! 😊"
            )
            yield "\n\n" + _json_s.dumps({"sources": [], "done": True, "needs_human": True})
            return
        else:
            # Use reformulated query as LLM question for detected follow-ups
            # ("give me in detail" → retrieval_query = "life insurance coverage details")
            # so the model sees the actual topic, not the vague follow-up phrase.
            # _detected_as_followup is used (not _is_followup) so the fix applies
            # even when we fell back to history-based topic extraction.
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
                    f"{prompt_question.rstrip(' .?')} — this may be a new question "
                    f"unrelated to the earlier conversation; if so, open your answer "
                    f"with something like \"If you're asking about {{topic}} generally, \" "
                    f"before answering, naming the actual topic instead of the placeholder."
                )

            # ── Modifier-signal instruction injection ─────────────────────────
            # Detect what kind of modifier the user asked for (example / simple /
            # detail) and inject a targeted instruction into prompt_question so
            # the LLM knows exactly what format/style is expected.
            _q_lower_mod = question.lower()
            _has_example = _wants_example(_q_lower_mod)
            _has_simple  = any(sig in _q_lower_mod for sig in _SIMPLE_SIGNALS)
            _has_detail  = any(sig in _q_lower_mod for sig in _DETAIL_SIGNALS)

            if _detected_as_followup and (_has_example or _has_simple or _has_detail):
                # Build instruction based on the combination of signals.
                if _has_detail and _has_simple and _has_example:
                    _mod_instr = (
                        "Give a detailed explanation in simple, everyday language with one "
                        "concrete real-life example. No jargon — explain like you would to a friend. "
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
                        "Re-explain in very simple, everyday language — no jargon. "
                        "Then give one concrete real-life example to illustrate it clearly."
                    )
                elif _has_example:
                    _mod_instr = (
                        "The user already understands this concept from the previous answer. "
                        "Do NOT re-explain or repeat the definition. "
                        "Give ONLY one concrete real-life example that illustrates it clearly."
                    )
                elif _has_simple:
                    _mod_instr = (
                        "Re-explain this in very simple, everyday language. "
                        "No jargon or technical terms — plain words only."
                    )
                elif _has_detail:
                    _mod_instr = (
                        "Give a full, detailed breakdown. "
                        "The user wants more depth — use numbered points."
                    )
                else:
                    _mod_instr = ""
                if _mod_instr:
                    prompt_question = f"{prompt_question.rstrip(' .?')} — {_mod_instr}"
            elif not _detected_as_followup and _has_example:
                # Fresh question asking for an explanation with an example
                prompt_question = (
                    f"{prompt_question.rstrip(' .?')} — "
                    "please include a concrete real-life example to illustrate this concept clearly."
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

        # ── Trust gate: don't live-stream when content won't be trusted ───
        # _top_rerank < 0.05 means the answer would be discarded if a Rule 4
        # marker appears (see _r4_trusted below). Skip live token-by-token
        # streaming in this case — buffer the full response, strip any
        # refusal marker, THEN yield the clean result — so the user never
        # sees bad text even momentarily. The non-streaming fallback block
        # below already implements exactly this "generate fully, then strip,
        # then yield" flow, so we just force it to run by leaving
        # streamed_ok = False.
        _trust_rerank = _top_rerank

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

        # When the reranker confidence is low (_top_rerank < 0.05), skip
        # live token-by-token streaming entirely. The non-streaming fallback
        # below generates the full response, strips any Rule4 refusal marker,
        # and THEN yields the clean text — so the user never sees bad text
        # even momentarily. For high-confidence queries, stream live as before.
        if _stream_url and _trust_rerank >= 0.05:
            model = _stream_model
            url = _stream_url
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(__import__("os").getenv("VLLM_MAX_TOKENS", "600") if detailed else __import__("os").getenv("VLLM_MAX_TOKENS_BRIEF", "300")),
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
                                    token = _json.loads(data)["choices"][0]["delta"].get("content", "") or ""
                                    if token:
                                        _kv_reply += token
                                        yield token
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

        # The prompt explicitly says the filler word "honestly" must NEVER be
        # shortened to "honest," — the model doesn't reliably follow that
        # (confirmed live, same call, non-deterministic: "Honestly, it's..."
        # on one run, "Honest, it's..." on the next, identical question and
        # context). Same lesson as the Rule4/truncation fixes right below:
        # don't trust a prompt instruction the model won't consistently
        # honor — enforce it deterministically instead. Only matches the
        # filler-word usage (comma immediately after "honest"), not a
        # legitimate adjective use ("an honest answer").
        _honest_fixed = _HONEST_FILLER_RE.sub(lambda m: f"{m.group(1)}onestly,", _reply_stripped)
        if _honest_fixed != _reply_stripped:
            _reply_stripped = _honest_fixed
            _kv_reply = _honest_fixed
            _corrected_text = _honest_fixed

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
                "Let me get one of our agents on it — they'll be able to help you better! 😊"
            )
            _reply_stripped = _refusal_text
            _kv_reply = _refusal_text
            _corrected_text = _refusal_text
            _rule4_discarded = True
            logger.info(
                "[ask_stream] discarded low-confidence Rule4 answer (top_rerank=%.3f < 0.05)",
                _top_rerank,
            )

        # ── Truncation detection ─────────────────────────────────────────────
        # If the stream hit max_tokens mid-sentence, trim to the last complete
        # sentence and tell the frontend to replace the displayed text.
        _SENT_ENDERS = frozenset('.!?…')
        if _reply_stripped and _reply_stripped[-1] not in _SENT_ENDERS:
            # Response doesn't end at a sentence boundary — find the last one
            _last_sent = max(
                _reply_stripped.rfind('. '),
                _reply_stripped.rfind('! '),
                _reply_stripped.rfind('? '),
                _reply_stripped.rfind('.\n'),
                _reply_stripped.rfind('!\n'),
                _reply_stripped.rfind('?\n'),
            )
            if _last_sent > len(_reply_stripped) // 3:
                _corrected_text = _reply_stripped[:_last_sent + 1].strip()
                _kv_reply = _corrected_text  # cache the clean version

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
        # Wrapped in try/except so any edge-case failure keeps the original reply.
        if not _keyword_detailed and not _kv_has_example:
            try:
                import re as _re
                _cap_src = (_corrected_text or _reply_stripped).strip()
                if _cap_src:
                    # Simple split: find positions of sentence-ending punctuation
                    # followed by whitespace, then take first N chunks.
                    _sent_parts = _re.split(r'(?<=[.!?])\s+', _cap_src)
                    _MAX_SENTENCES = 6
                    if len(_sent_parts) > _MAX_SENTENCES:
                        _capped = " ".join(_sent_parts[:_MAX_SENTENCES]).strip()
                        # Ensure it ends cleanly
                        if _capped and _capped[-1] not in '.!?':
                            _capped += '.'
                        _corrected_text = _capped
                        _kv_reply = _capped
            except Exception as _cap_exc:
                logger.debug("[ask_stream] sentence cap skipped: %s", _cap_exc)

        # ── Buffered-path single yield (after all corrections) ────────────────
        # The non-streaming (buffered) path above stored the raw answer in
        # _kv_reply but did NOT yield it — we waited until after Rule4 strip,
        # truncation detection, and sentence cap so the client never sees
        # un-vetted text even momentarily.  Yield exactly once, only for the
        # buffered path; the live-streaming path already yielded tokens as
        # they arrived and must NOT hit this yield.
        if not streamed_ok:
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
        # Suggestions are generated once here (not in the KV-hit branch above)
        # and cached alongside the answer, so a cache hit on a later, identical
        # question still gets chips instead of silently having none — the
        # KV-hit early-return never re-derives full_context, so without this
        # the suggestion pipeline would only ever run on the first ask.
        _reply_for_chips = (_corrected_text or _kv_reply or "").strip()
        _sugg: list = []
        if _reply_for_chips and not any(p in _reply_for_chips.lower() for p in _handoff_phrases):
            _sugg = await _generate_suggestions(question, _reply_for_chips, context=full_context)

        if _kv_reply and unique_sources and not any(p in _kv_reply.lower() for p in _handoff_phrases):
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
                        "suggested_questions": _sugg,
                    },
                    query_embedding=_kv_q_emb,
                    query_text=retrieval_query,
                )
                logger.info("[ask_stream] KV cache stored for query=%r", retrieval_query[:80])
            except Exception as _exc:
                logger.debug("[ask_stream] KV cache write failed: %s", _exc)

        _final_payload: dict = {
            "sources": [] if _rule4_discarded else unique_sources,
            "done": True,
        }
        if _rule4_discarded:
            _final_payload["needs_human"] = True
        if _corrected_text:
            _final_payload["corrected_text"] = _corrected_text
        if _sugg:
            _final_payload["suggested_questions"] = _sugg
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