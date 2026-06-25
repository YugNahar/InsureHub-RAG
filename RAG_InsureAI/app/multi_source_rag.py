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

_LLM_BACKEND_ERRORS = (_APIConnectionError, _APITimeoutError, _APIStatusError)

# ── Context coverage check ────────────────────────────────────────────────────
# Domain-generic terms that appear in virtually every insurance chunk and
# therefore give no signal about whether a chunk covers the specific query topic.
_QUERY_STOP_WORDS = {
    'what', 'is', 'are', 'the', 'a', 'an', 'in', 'of', 'for', 'how', 'does',
    'do', 'i', 'my', 'me', 'by', 'with', 'under', 'about', 'can', 'will',
    'which', 'when', 'where', 'to', 'and', 'or', 'at', 'this', 'that', 'it',
    'its', 'be', 'been', 'has', 'have', 'had', 'any', 'all', 'from', 'on',
    # domain-generic insurance terms that appear in almost every chunk
    'insurance', 'insured', 'insurer', 'policy', 'policies', 'cover', 'coverage',
    'plan', 'claim', 'claims', 'benefits', 'benefit',
    # generic time/measurement words — too common to indicate topic coverage
    'period', 'time', 'duration', 'date', 'days', 'year', 'month', 'months',
}

def _context_covers_query(query: str, docs: list) -> bool:
    """True if ≥1 retrieved chunk contains at least one discriminating query keyword.

    Guards against cross-encoder false positives where a chunk scores highly but
    doesn't actually contain any query-specific term (e.g. a 'Duty of Assured'
    chunk scoring 0.87 for 'What is a deductible?').
    """
    query_terms = {
        w for w in re.findall(r'\b[a-z]{3,}\b', query.lower())
        if w not in _QUERY_STOP_WORDS
    }
    if not query_terms:
        return True  # no discriminating terms — assume covered
    for doc in docs:
        text = (
            doc.page_content if hasattr(doc, 'page_content') else doc.get('text', '')
        ).lower()
        chunk_terms = set(re.findall(r'\b[a-z]{3,}\b', text))
        # Require at least 3 overlapping keywords — 2 was too loose and let
        # tangentially related chunks pass to the LLM, which then answered from
        # training knowledge instead of refusing.
        if len(query_terms & chunk_terms) >= 3:
            return True
    return False


_HANDOFF_MSG = (
    "I don't have that in my knowledge base right now — "
    "let me get a human agent to help you! 😊"
)

# Phrases that indicate the model answered from general training knowledge
# rather than from the retrieved context. If ANY of these appear in the
# LLM response, we discard the response and return the handoff message.
_GENERAL_KNOWLEDGE_TELLS = [
    "generally speaking",
    "in general,",
    "typically,",
    "typically ",
    "usually,",
    "usually ",
    "as a general rule",
    "in most cases",
    "commonly,",
    "commonly ",
    "standard practice",
    "based on my knowledge",
    "from my training",
    "most insurance policies",
    "most insurers",
    "it is important to note",
    "one should ",
    "you should note",
    "please note that",
]


def _llm_used_general_knowledge(response: str) -> bool:
    """Return True if the LLM response contains general-knowledge tells."""
    lower = response.lower()
    return any(tell in lower for tell in _GENERAL_KNOWLEDGE_TELLS)


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
    # Remove ATX headers (## Heading → Heading)
    text = re.sub(r'^#{1,4}\s+', '', text, flags=re.MULTILINE)
    # Convert bullet list items to flowing prose: "- item" or "* item" → "item, "
    # First item on a new line: replace "\n- " with ", "
    text = re.sub(r'\n\s*[-*]\s+', ' ', text)
    # Numbered list items: "\n1. " → " "
    text = re.sub(r'\n\s*\d+\.\s+', ' ', text)
    # Inline code backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Collapse 3+ newlines → 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Strip leftover leading/trailing whitespace per line
    lines = [l.rstrip() for l in text.split('\n')]
    return '\n'.join(lines)


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
        r"\btranslate\b|\bmeaning\s+of\b"
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
from prompt_template import STRICT_GROUNDED_PROMPT, CALCULATION_PROMPT, CONVERSATIONAL_RAG_PROMPT
from context_compressor import ContextCompressor
from rag import LLM_CONTEXT_WINDOW_CHARS

logger = logging.getLogger(__name__)

class MultiSourceRAG:
    def __init__(self):
        self.doc_pipeline = RAGPipeline()
        self.video_store = VideoVectorStore()
        self.webpage_store = WebpageVectorStore()
        self.max_context_chars = 4000   # ~8 × 500-char semantic chunks
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
            if h not in seen or chunk.metadata.get("similarity", 0) > seen[h].metadata.get("similarity", 0):
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
                in_pool_srcs = {d.metadata.get("source", "") for d in doc_chunks}
                for summary_doc in relevant_summaries:
                    src = summary_doc.metadata.get("source", "")
                    if src and src not in in_pool_srcs:
                        boost = await asyncio.to_thread(
                            self.doc_pipeline._vector_store.search,
                            retrieval_query, 2, {"source": {"$eq": src}}, True, False,
                        )
                        if boost:
                            existing = {d.page_content[:80] for d in doc_chunks}
                            doc_chunks = doc_chunks + [
                                d for d in boost if d.page_content[:80] not in existing
                            ]
                            in_pool_srcs.add(src)
                            logger.info(
                                "[MultiSourceRAG] stage-1 boost: added %d chunk(s) from %r",
                                len(boost), src,
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
        # If the top chunk has similarity <= 0.30, the retrieval found nothing
        # meaningfully relevant (typical unrelated content scores 0.1–0.3).
        # Flag needs_human so the caller can trigger a human handoff.
        # 
        # Short follow-ups with available history bypass raw-similarity detection:
        # the reformulated query should retrieve meaningful context, and even if
        # scores are low the conversation history provides enough grounding.
        top_similarity = all_chunks[0].metadata.get("similarity", 0) if all_chunks else 0
        needs_human = (top_similarity <= 0.30)
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

        # Always use STRICT_GROUNDED_PROMPT — CONVERSATIONAL_RAG_PROMPT with a
        # small 7B model is unreliable at honouring "no training knowledge" rules.
        prompt = STRICT_GROUNDED_PROMPT.format(history=history, context=full_context, question=question)
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
        if _llm_used_general_knowledge(answer):
            logger.info("[MultiSourceRAG] LLM used general knowledge — replacing with handoff message")
            answer = _HANDOFF_MSG
            needs_human = True
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

        # Streaming path uses a pool of 8 candidates (vs 30 in ask()) so the
        # cross-encoder rerank takes ~2-3s on CPU instead of ~9s, keeping
        # first-token latency under 5s while still reordering by relevance.
        # ── Parallel retrieval across all sources ─────────────────────────────
        if not document_filter:
            doc_chunks, video_chunks, webpage_chunks = await asyncio.gather(
                self._retrieve_doc_chunks(retrieval_query, filter_meta, document_filter, doc_top_k=8, summary_top_k=3),
                asyncio.to_thread(self.video_store.search, retrieval_query, top_k=4, use_hybrid=True, use_reranker=False),
                asyncio.to_thread(self.webpage_store.search, retrieval_query, top_k=4, use_hybrid=True, use_reranker=False),
            )
            all_chunks = self._merge_chunks(doc_chunks + video_chunks + webpage_chunks)
        else:
            doc_chunks = await self._retrieve_doc_chunks(retrieval_query, filter_meta, document_filter, doc_top_k=8, summary_top_k=3)
            all_chunks = self._merge_chunks(doc_chunks)

        all_chunks.sort(key=lambda x: x.metadata.get("similarity", 0), reverse=True)
        all_chunks = all_chunks[:8]

        total_retrieved_chars = sum(len(c.page_content) for c in all_chunks)
        if total_retrieved_chars > LLM_CONTEXT_WINDOW_CHARS:
            all_chunks = self._compressor.compress_to_budget(
                question, all_chunks, max_total_chars=LLM_CONTEXT_WINDOW_CHARS
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
        if len(full_context) > self.max_context_chars:
            full_context = full_context[:self.max_context_chars] + "... (truncated)"

        if not full_context.strip():
            yield (
                "Hmm, I don't have that specific information in my knowledge base right now. "
                "Let me get one of our agents on it — they'll be able to help you better! 😊"
            )
            return
        else:
            ctx_covered = _context_covers_query(question, all_chunks)
            if not ctx_covered and not document_filter:
                yield (
                    "Hmm, I don't have that specific information in my knowledge base right now. "
                    "Let me get one of our agents on it — they'll be able to help you better! 😊"
                )
                return
            prompt = STRICT_GROUNDED_PROMPT.format(history=history, context=full_context, question=question)
            llm = get_insurance_llm(temperature=0)

        # ── Stream LLM tokens directly via vLLM HTTP SSE ─────────────────────
        # LangChain's astream() buffers the full response before yielding.
        # We bypass it and call the vLLM /v1/chat/completions endpoint with
        # stream=True so the frontend sees the first word in <1 second.
        import json as _json
        import aiohttp
        from router import VLLM_HOST, VLLM_API_KEY, _resolve_vllm_model, _active_backend

        unique_sources = list(dict.fromkeys(sources))
        streamed_ok = False

        if _active_backend() == "vllm" and VLLM_HOST:
            model = _resolve_vllm_model()
            url = f"{VLLM_HOST}/v1/chat/completions"
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": int(__import__("os").getenv("VLLM_MAX_TOKENS", "1024")),
                "stream": True,
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VLLM_API_KEY}",
            }
            try:
                buffered_tokens: list[str] = []
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                        async for raw_line in resp.content:
                            line = raw_line.decode().strip()
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                token = _json.loads(data)["choices"][0]["delta"].get("content", "")
                                if token:
                                    buffered_tokens.append(token)
                            except Exception:
                                pass
                full_streamed = "".join(buffered_tokens)
                if _llm_used_general_knowledge(full_streamed):
                    logger.info("[ask_stream] LLM used general knowledge — replacing with handoff")
                    yield _HANDOFF_MSG
                else:
                    for token in buffered_tokens:
                        yield token
                streamed_ok = True
            except Exception as exc:
                logger.warning("[ask_stream] direct vLLM streaming failed, falling back: %s", exc)

        if not streamed_ok:
            # Fallback: regular invoke (no streaming)
            response = await asyncio.to_thread(llm.invoke, prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            answer = _strip_markdown(_strip_model_preamble(answer))
            if _llm_used_general_knowledge(answer):
                logger.info("[ask_stream] fallback LLM used general knowledge — replacing with handoff")
                answer = _HANDOFF_MSG
            yield answer

        yield "\n\n" + _json.dumps({"sources": unique_sources, "done": True})

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