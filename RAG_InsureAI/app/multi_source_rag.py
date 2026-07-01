"""
Unified RAG with strict grounding – no hallucinations.
Supports document filtering with substring matching.
"""
import asyncio
import logging
import re
from typing import List, Tuple, Optional

try:
    from openai import APIConnectionError as _APIConnectionError, APITimeoutError as _APITimeoutError, APIStatusError as _APIStatusError
except ImportError:
    _APIConnectionError = Exception
    _APITimeoutError = Exception
    _APIStatusError = Exception

from rapidfuzz import fuzz, process

_LLM_BACKEND_ERRORS = (_APIConnectionError, _APITimeoutError, _APIStatusError)

# ── Typo-tolerant insurance vocabulary ──────────────────────────────────────────
# Correctly-spelled insurance-domain terms used by _correct_typos() to
# fix common typos before vector retrieval.
_INSURANCE_VOCAB = [
    "insurance", "policy", "premium", "deductible", "coverage", "claim", "claims",
    "insured", "insurer", "underwriting", "underwrite", "renewal", "nominee",
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


def _correct_typos(text: str) -> str:
    """Fix common typos in insurance-domain terms using fuzzy matching.

    Splits *text* into words. For each word of length >= 4, finds the best
    match in ``_INSURANCE_VOCAB`` using ``rapidfuzz`` with ``fuzz.ratio``.
    If the best match score >= 80 and is not already exact, replaces the
    word with the correctly-spelled vocabulary term. Returns the corrected
    sentence with original word order preserved.
    """
    words = text.split()
    corrected = []
    for w in words:
        stripped = w.strip(".,!?;:()[]{}'\"")
        if len(stripped) >= 4:
            result = process.extractOne(stripped, _INSURANCE_VOCAB, scorer=fuzz.ratio)
            if result is not None:
                best_match, score, _ = result
                if score >= 80 and best_match != stripped:
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
    # Domain-generic insurance terms that appear in almost every chunk
    'insurance', 'insured', 'insurer', 'policy', 'policies', 'cover', 'coverage',
    'plan', 'claim', 'claims', 'benefits', 'benefit',
    # Generic time/measurement words — too common to indicate topic coverage
    'period', 'time', 'duration', 'date', 'days', 'year', 'month', 'months',
}


def _extract_topic_terms(query: str) -> set[str]:
    """Return discriminating topic words from a query after stop-word filtering."""
    return {
        w for w in re.findall(r'\b[a-z]{3,}\b', query.lower())
        if w not in _QUERY_STOP_WORDS
    }


def _word_matches(topic: str, chunk_word: str) -> bool:
    """True if topic and chunk_word are the same root.

    Uses a 5-character shared-prefix heuristic to handle common inflections
    without a full stemmer, e.g.:
      "deductible" / "deduction" / "deducted"  → share "deduct" ✓
      "nominate"   / "nomination"               → share "nomin"  ✓
      "tax"        / "taxation"                 → share "tax"    ✓ (exact)
    """
    if topic == chunk_word:
        return True
    # Require at least 5 chars in both to avoid spurious short-word matches
    min_len = 5
    prefix = min(len(topic), len(chunk_word), min_len)
    return prefix >= min_len and topic[:prefix] == chunk_word[:prefix]


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
        # Build a combined term set across ALL retrieved chunks.
        all_terms: set[str] = set()
        for doc in docs:
            text = (
                doc.page_content if hasattr(doc, 'page_content') else doc.get('text', '')
            ).lower()
            all_terms.update(re.findall(r'\b[a-z]{3,}\b', text))

        # Also pass if the top chunk has a high rerank score — the cross-encoder
        # has validated relevance and the keyword check should not override it.
        top_rerank = max(
            (d.metadata.get("rerank_score", 0) for d in docs if hasattr(d, "metadata")),
            default=0,
        )
        if top_rerank >= 0.5:
            return True

        # OR-logic: pass if AT LEAST ONE topic phrase is fully covered.
        # The original AND-logic (all topics must match) was too strict —
        # a single missing synonym killed answers that were clearly in the KB.
        for topic in llm_topics:
            words = [w for w in topic.replace('-', ' ').split() if len(w) >= 3]
            if not words:
                continue
            if all(any(_word_matches(w, c) for c in all_terms) for w in words):
                return True  # at least one topic fully covered → adequate context
        return False

    # Regex fallback: split compound questions on "and"/"or".
    sub_queries = [q.strip() for q in re.split(r'\band\b|\bor\b', query, flags=re.IGNORECASE) if q.strip()]
    all_term_sets = [_extract_topic_terms(sq) for sq in sub_queries]
    all_term_sets = [t for t in all_term_sets if t]
    if not all_term_sets:
        return True

    for doc in docs:
        text = (
            doc.page_content if hasattr(doc, 'page_content') else doc.get('text', '')
        ).lower()
        chunk_terms = set(re.findall(r'\b[a-z]{3,}\b', text))
        for topic_terms in all_term_sets:
            if _topics_hit_chunk(topic_terms, chunk_terms):
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
}


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


async def _llm_classify_intent(question: str) -> bool:
    """Classify whether a question needs a detailed answer (True) or brief (False).

    Fast path: keyword signals handle the obvious cases with zero latency.
    LLM path: a non-blocking aiohttp call (max 5 tokens, 5 s timeout) for
    ambiguous questions. Falls back to False (brief) on any error or timeout.
    Designed to run in parallel with retrieval — adds zero wall-clock latency.
    """
    q = question.lower()
    # Fast path — explicit simple/brief signal overrides everything
    if any(sig in q for sig in _SIMPLE_SIGNALS):
        return False
    # Fast path — unambiguously detailed
    if any(sig in q for sig in _DETAIL_SIGNALS):
        return True
    if len(question.split()) > 25:
        return True
    if q.count(' and ') >= 3:
        return True

    # LLM path — non-blocking aiohttp call, same pattern as _extract_intent_topics
    try:
        import aiohttp as _aiohttp
        from router import VLLM_HOST, VLLM_API_KEY, _resolve_vllm_model, _active_backend
        if _active_backend() != "vllm" or not VLLM_HOST:
            return False
        prompt = (
            "Classify this insurance question with ONE word only.\n"
            "Reply 'detailed' if it needs steps, a list, or multiple parts.\n"
            "Reply 'brief' if 2-3 sentences would fully answer it.\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        payload = {
            "model": _resolve_vllm_model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 5,
            "temperature": 0,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_API_KEY}",
        }
        timeout = _aiohttp.ClientTimeout(total=5)
        async with _aiohttp.ClientSession() as session:
            async with session.post(
                f"{VLLM_HOST}/v1/chat/completions",
                json=payload, headers=headers, timeout=timeout,
            ) as resp:
                data = await resp.json()
                result = data["choices"][0]["message"]["content"].strip().lower()
                return "detailed" in result
    except Exception:
        return False  # safe default: brief


_FOLLOWUP_SIGNALS = {
    'it', 'its', 'that', 'this', 'them', 'those', 'they', 'their', 'which', 'these',
}
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


async def _generate_suggestions(question: str, answer: str, context: str = "") -> list:
    """
    Generate follow-up chip questions grounded in the retrieved KB context.

    Steps:
      1. Ask the LLM to produce 5 candidate questions from the KB snippet.
      2. Verify each candidate by checking content-word overlap with the full
         retrieved context — questions whose key terms don't appear in the
         context are dropped before they reach the user.
      3. Return up to 3 verified questions.
    """
    try:
        import aiohttp as _ah
        from router import VLLM_HOST, VLLM_API_KEY, _resolve_vllm_model, _active_backend
        if _active_backend() != "vllm" or not VLLM_HOST:
            return []
        if not context or not context.strip():
            return []
        # Use a larger snippet so the model has enough material to draw from
        ctx_snippet = context[:1000].strip()
        prompt = (
            f"Knowledge base content:\n{ctx_snippet}\n\n"
            f"User asked: {question}\n\n"
            "Write 5 short follow-up questions (max 8 words each) that:\n"
            "1. Ask about specific facts, terms, or details mentioned in the knowledge base content above.\n"
            "2. A user would naturally ask AFTER reading the answer above.\n"
            "3. Can be answered ONLY from the knowledge base content above — do not invent topics.\n"
            "Output only the questions, one per line, no numbering, no bullets, no explanations:"
        )
        async with _ah.ClientSession() as _s:
            async with _s.post(
                f"{VLLM_HOST}/v1/chat/completions",
                json={
                    "model": _resolve_vllm_model(),
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 120,
                    "stream": False,
                },
                headers={"Authorization": f"Bearer {VLLM_API_KEY}"},
                timeout=_ah.ClientTimeout(total=6),
            ) as _r:
                if _r.status != 200:
                    return []
                _data = await _r.json()
                _text = _data["choices"][0]["message"]["content"].strip()
        # Parse and clean candidates
        _candidates = [
            q.strip().lstrip("0123456789.-) ").strip()
            for q in _text.split("\n") if q.strip()
        ]
        _candidates = [q for q in _candidates if 2 <= len(q.split()) <= 10]
        # Verify each candidate has actual coverage in the full retrieved context
        _verified = [q for q in _candidates if _question_answerable_in_context(q, context)]
        return _verified[:3]
    except Exception:
        pass
    return []


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

    return False


async def _reformulate_query(question: str, history: str) -> str:
    """Rewrite a follow-up question as a standalone search query using conversation history.

    Examples:
      history: "User: tell me about life insurance\nLayla: Life insurance ..."
      question: "what about premiums?"
      returns: "life insurance premiums"

    Uses max_tokens=30 and a 4-second timeout so it adds <0.5 s to latency.
    Falls back to the original question on any error.
    """
    import aiohttp as _aiohttp
    from router import VLLM_HOST, VLLM_API_KEY, _resolve_vllm_model, _active_backend
    if _active_backend() != "vllm" or not VLLM_HOST:
        return question
    # Use only the last 6 lines (3 turns) of history to keep the prompt short
    recent = '\n'.join(history.strip().split('\n')[-6:])
    prompt = f"""Rewrite the follow-up question as a short standalone search query using the conversation context.
Use precise insurance/legal terms that would appear in a textbook (not casual phrasing).
Output ONLY the search query — no quotes, no explanation, nothing else.

Examples:
  Context: "User: tell me about life insurance\nLayla: Life insurance pays out..."
  Follow-up: "what about premiums?" → "life insurance premium amount"

  Context: "User: explain life insurance\nLayla: Life insurance protects your family..."
  Follow-up: "is it tax deductible?" → "life insurance premiums tax deductible section 80C income"

  Context: "User: explain reinsurance\nLayla: Reinsurance is when insurers share risk..."
  Follow-up: "is it legally required?" → "reinsurance legal requirement regulation"

  Context: "User: what is subrogation\nLayla: Subrogation means the insurer steps in..."
  Follow-up: "give me an example" → "subrogation example real case"

  Context: "User: what is a deductible\nLayla: A deductible is what you pay first..."
  Follow-up: "how is it calculated?" → "deductible calculation formula percentage"

Conversation:
{recent}

Follow-up: {question}
Search query:"""
    try:
        payload = {
            "model": _resolve_vllm_model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30,
            "temperature": 0,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_API_KEY}",
        }
        async with _aiohttp.ClientSession() as session:
            async with session.post(
                f"{VLLM_HOST}/v1/chat/completions",
                json=payload, headers=headers,
                timeout=_aiohttp.ClientTimeout(total=4),
            ) as resp:
                data = await resp.json()
                reformulated = data["choices"][0]["message"]["content"].strip().strip('"\'')
                if reformulated and len(reformulated) >= 3:
                    logger.info("[REFORM] %r -> %r", question, reformulated)
                    return reformulated
    except Exception as exc:
        logger.debug("[REFORM] failed (%s) — using original question", exc)
    return question


_INTENT_PROMPT = """\
Extract the specific insurance topic from the question below.
Output ONLY the topic as 1-5 comma-separated words or short phrases — nothing else.
Keep compound terms together (e.g. "no-claim bonus", "free look period", "sum assured").

Examples:
"what is life insurance"                                       → life insurance
"what is a no-claim bonus"                                     → no-claim bonus
"what is a free look period"                                   → free look period
"what is a deductible"                                         → deductible
"explain reinsurance"                                          → reinsurance
"how are premiums calculated for fire insurance"               → fire insurance, premium
"what is the difference between term and whole life"           → term life, whole life
"how to file a motor insurance claim"                          → motor claim
"what documents are needed for health insurance claim"         → health claim, documents
"how does reinsurance work and what is its legal significance" → reinsurance, legal
"tell me about motor policy claims"                            → motor, claims

Question: {question}
Topic:"""


async def _extract_intent_topics(question: str) -> set[str]:
    """Fast LLM call (runs in parallel with retrieval) to extract core topic words.

    Uses vLLM with max_tokens=30 so the round-trip is <1 s on an idle server.
    Falls back to an empty set on any error — caller uses regex fallback.
    """
    import aiohttp as _aiohttp
    from router import VLLM_HOST, VLLM_API_KEY, _resolve_vllm_model, _active_backend
    if _active_backend() != "vllm" or not VLLM_HOST:
        return set()
    try:
        prompt = _INTENT_PROMPT.format(question=question)
        payload = {
            "model": _resolve_vllm_model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 30,
            "temperature": 0,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_API_KEY}",
        }
        timeout = _aiohttp.ClientTimeout(total=5)
        async with _aiohttp.ClientSession() as session:
            async with session.post(f"{VLLM_HOST}/v1/chat/completions",
                                    json=payload, headers=headers,
                                    timeout=timeout) as resp:
                data = await resp.json()
                raw = data["choices"][0]["message"]["content"].strip().lower()
                # Parse phrases like "no-claim bonus, deductible" keeping
                # multi-word/hyphenated compounds intact for phrase matching.
                topics: set[str] = set()
                for phrase in raw.split(","):
                    phrase = re.sub(r"[^a-z\- ]", "", phrase).strip()
                    if phrase:
                        topics.add(phrase)  # keep compound e.g. "no-claim bonus"
                logger.debug("[INTENT] %r → topics=%s", question, topics)
                return topics
    except Exception as exc:
        logger.debug("[INTENT] extraction failed (%s) — using regex fallback", exc)
        return set()


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


def _reformulate_with_history(question: str, history: str) -> str:
    """Merge a short follow-up with the last assistant turn from *history*.

    The returned string is used **only** for the retrieval query — the original
    *question* is still passed to the LLM prompt so the model answers what the
    user actually asked.
    """
    lines = history.strip().split("\n")
    last_assistant = ""
    for line in reversed(lines):
        if line.startswith("Assistant:"):
            last_assistant = line[len("Assistant:"):].strip()
            break
    if last_assistant:
        return f"{last_assistant} {question}"
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

logger = logging.getLogger(__name__)

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
                    boost = await asyncio.to_thread(
                        self.doc_pipeline._vector_store.search,
                        retrieval_query, 5, {"source": {"$eq": src}}, True, False,
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
        ctx_covered = _context_covers_query(question, all_chunks)

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
            llm = get_insurance_llm(temperature=0.3)

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

        # ── Pure conversational replies — no retrieval needed ─────────────────
        # "yes", "no", "ok", "thanks" etc. have zero retrieval value.
        _PURE_CONV = frozenset({
            "yes", "no", "ok", "okay", "sure", "alright", "nope", "nah",
            "thanks", "thank you", "got it", "i see", "understood", "right",
            "cool", "great", "nice", "fine", "good", "perfect", "awesome",
            "no thanks", "no thank you", "not really", "never mind", "nevermind",
        })
        _q_stripped = question.strip().lower().strip("!.,?;:")
        if _q_stripped in _PURE_CONV:
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
                # LLM reformulation failed (timeout / vLLM busy) — fall back to the
                # last user question from history so "give me in detail" maps to the
                # actual topic instead of colliding with every other detail request.
                _last_user_q = next(
                    (ln[len("User:"):].strip() for ln in reversed(history.split("\n"))
                     if ln.startswith("User:")),
                    None,
                )
                if _last_user_q and _last_user_q.lower() != question.lower():
                    retrieval_query = _last_user_q
                    _is_followup = True
        else:
            retrieval_query = corrected_question

        # ── Keyword detailed check — must run BEFORE cache ───────────────────
        # So the cache key correctly separates brief vs. detailed for the same
        # topic ("what is health insurance" vs "explain health insurance in detail").
        _keyword_detailed = _needs_detailed_answer(question)
        _doc_top_k   = 14 if _keyword_detailed else 8
        _chunk_limit = 12 if _keyword_detailed else 8
        _media_top_k =  5 if _keyword_detailed else 4

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
        _kv_has_example = any(sig in _q_lower_intent for sig in _EXAMPLE_SIGNALS)
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
            yield _kv_hit.get("answer", "")
            yield "\n\n" + _json_s.dumps({
                "sources": _kv_hit.get("sources", []),
                "done": True,
                "needs_human": False,
            })
            return

        # Streaming path: parallel retrieval + LLM intent extraction + LLM intent classification.
        # _keyword_detailed is already computed above; OR with LLM result for final flag.
        if not document_filter:
            (doc_chunks, video_chunks, webpage_chunks), llm_topics, _llm_detailed = await asyncio.gather(
                asyncio.gather(
                    self._retrieve_doc_chunks(retrieval_query, filter_meta, document_filter, doc_top_k=_doc_top_k, summary_top_k=3),
                    asyncio.to_thread(self.video_store.search, retrieval_query, top_k=_media_top_k, use_hybrid=True, use_reranker=False),
                    asyncio.to_thread(self.webpage_store.search, retrieval_query, top_k=_media_top_k, use_hybrid=True, use_reranker=False),
                ),
                _extract_intent_topics(question),
                _llm_classify_intent(question),
            )
            all_chunks = self._merge_chunks(doc_chunks + video_chunks + webpage_chunks)
        else:
            doc_chunks, llm_topics, _llm_detailed = await asyncio.gather(
                self._retrieve_doc_chunks(retrieval_query, filter_meta, document_filter, doc_top_k=_doc_top_k, summary_top_k=3),
                _extract_intent_topics(question),
                _llm_classify_intent(question),
            )
            all_chunks = self._merge_chunks(doc_chunks)
        # Only trigger detailed mode on explicit user signals ("in detail", "step by step", etc.)
        # _llm_detailed is intentionally excluded — the LLM over-classifies insurance questions
        # (which often have lists/procedures) as "detailed", making every answer verbose.
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

        # Run coverage check on the pre-compression chunks so that video/webpage
        # chunks filling the budget first don't cause doc-chunk terms to go missing.
        import json as _json_s
        topics_for_coverage = None if _detected_as_followup else (llm_topics or None)
        ctx_covered = _context_covers_query(retrieval_query, all_chunks, llm_topics=topics_for_coverage)
        if not ctx_covered and not document_filter:
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
            # If user asked for an example, append explicit instruction so the LLM
            # generates a concrete illustrative scenario even if not in the KB.
            _q_lower = question.lower()
            if any(sig in _q_lower for sig in _EXAMPLE_SIGNALS):
                prompt_question = (
                    prompt_question.rstrip(" .?") +
                    " — please include a concrete real-life example to illustrate this concept clearly."
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
                llm = get_insurance_llm(temperature=0.3)

        # ── Stream LLM tokens directly via vLLM HTTP SSE ─────────────────────
        # LangChain's astream() buffers the full response before yielding.
        # We bypass it and call the vLLM /v1/chat/completions endpoint with
        # stream=True so the frontend sees the first word in <1 second.
        import json as _json
        import aiohttp
        from router import VLLM_HOST, VLLM_API_KEY, _resolve_vllm_model, _active_backend

        unique_sources = list(dict.fromkeys(sources))
        streamed_ok = False
        _kv_reply = ""  # buffer for cache write

        if _active_backend() == "vllm" and VLLM_HOST:
            model = _resolve_vllm_model()
            url = f"{VLLM_HOST}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(__import__("os").getenv("VLLM_MAX_TOKENS", "600") if detailed else __import__("os").getenv("VLLM_MAX_TOKENS_BRIEF", "300")),
                "stream": True,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VLLM_API_KEY}",
            }
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
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
                logger.warning("[ask_stream] direct vLLM streaming failed, falling back: %s", exc)

        if not streamed_ok:
            # Fallback: regular invoke (no streaming)
            response = await asyncio.to_thread(llm.invoke, prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            answer = _strip_markdown(_strip_model_preamble(answer))
            _kv_reply = answer
            yield answer

        # ── Truncation detection ─────────────────────────────────────────────
        # If the stream hit max_tokens mid-sentence, trim to the last complete
        # sentence and tell the frontend to replace the displayed text.
        _reply_stripped = (_kv_reply or "").rstrip()
        _corrected_text = None
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
        # Wrapped in try/except so any edge-case failure keeps the original reply.
        if not _keyword_detailed:
            try:
                import re as _re
                _cap_src = (_corrected_text or _reply_stripped).strip()
                if _cap_src:
                    # Simple split: find positions of sentence-ending punctuation
                    # followed by whitespace, then take first 4 chunks.
                    _sent_parts = _re.split(r'(?<=[.!?])\s+', _cap_src)
                    _MAX_SENTENCES = 4
                    if len(_sent_parts) > _MAX_SENTENCES:
                        _capped = " ".join(_sent_parts[:_MAX_SENTENCES]).strip()
                        # Ensure it ends cleanly
                        if _capped and _capped[-1] not in '.!?':
                            _capped += '.'
                        _corrected_text = _capped
                        _kv_reply = _capped
            except Exception as _cap_exc:
                logger.debug("[ask_stream] sentence cap skipped: %s", _cap_exc)

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
                    },
                    query_embedding=_kv_q_emb,
                    query_text=retrieval_query,
                )
                logger.info("[ask_stream] KV cache stored for query=%r", retrieval_query[:80])
            except Exception as _exc:
                logger.debug("[ask_stream] KV cache write failed: %s", _exc)

        _final_payload: dict = {"sources": unique_sources, "done": True}
        if _corrected_text:
            _final_payload["corrected_text"] = _corrected_text
        _reply_for_chips = (_corrected_text or _kv_reply or "").strip()
        if _reply_for_chips and not any(p in _reply_for_chips.lower() for p in _handoff_phrases):
            _sugg = await _generate_suggestions(question, _reply_for_chips, context=full_context)
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