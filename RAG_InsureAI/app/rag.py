"""
Insurance RAG Pipeline — TurboVec Vector Edition with HyDE, Hybrid Search, and Citation Enforcement.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import uuid
from typing import Any, Optional

import pandas as pd
import requests
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from document_loader import load_document, load_url, extract_urls
from semantic_chunker import SemanticChunker
from metadata_tagger import (
    tag_document,
    classify_query,
    build_metadata_filter,
    classify_document_type,
    classify_chunk_intent,
    classify_chunk_policy_type,
)
from validator import detect_conflict, validate_grounding
from router import get_insurance_llm, get_general_llm, VLLM_HOST
from prompt_template import (
    GENERAL_PROMPT,
    RAG_PROMPT,
)
from vector_store import ChromaVectorStore
from summary_store import SummaryStore
from summarizer import generate_summary
from context_compressor import ContextCompressor
from kv_cache import QueryKVCache

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
RETRIEVE_K = 16
RERANK_K = 8
MAX_CONTEXT_CHARS = 10000
SUMMARY_MAX_CHARS = 20000
# How many candidate sources Stage 1 (summary search) hands to Stage 2
# (chunk search) in knowledge_query()'s two-stage retrieval.
SUMMARY_STAGE1_TOP_K = 5
LLM_CONTEXT_WINDOW_CHARS = 9000  # ~2250 tokens of context; prompt template ~300 + output 600 = ~3150 total, under Qwen's 4096

# ══════════════════════════════════════════════════════════════════════════════
# SECTION DETECTION (doc-type aware)
# ══════════════════════════════════════════════════════════════════════════════

_POLICY_SECTION_PATTERNS: dict[str, list[str]] = {
    "definitions": [
        r"\bdefin", r"\bmeans?\b", r"\bshall mean\b", r"\brefers? to\b",
        r"\binterpretation\b", r"\bglossary\b",
    ],
    "eligibility": [
        r"\beligib", r"\bminimum age\b", r"\bmaximum age\b", r"\bage limit\b",
        r"\bentry age\b", r"\binsured person\b", r"\bwho (can|may|is)\b",
        r"\bqualif", r"\brequirement\b", r"\bage of\b",
    ],
    "benefits": [
        r"\bbenefit\b", r"\bcoverage\b", r"\bcovers?\b", r"\bcompensation\b",
        r"\breimbursement\b", r"\bpayable\b", r"\blimit\b",
        r"\bsum insured\b", r"\bpayout\b", r"\bindemnity\b",
        r"\bmaximum benefit\b", r"\bschedule of benefit\b",
    ],
    "exclusions": [
        r"\bexclusion\b", r"\bnot cover", r"\bnot include", r"\bexclud",
        r"\bexcept\b", r"\bnot payable\b", r"\bvoid\b",
    ],
    "claims": [
        r"\bclaim\b", r"\bnotif", r"\bprocedure\b",
        r"\bsubmit\b", r"\bdocuments? required\b", r"\bfile a claim\b",
    ],
    "flight_delay": [
        r"\bflight delay\b", r"\btrip delay\b", r"\bdeparture delay\b",
        r"\bconsecutive hours?\b", r"\bhours?\s+delay\b", r"\bdelay benefit\b",
        r"\bdelay compensation\b", r"\btravel delay\b", r"\bflight delay benefit\b",
    ],
    "medical": [
        r"\bmedical expense", r"\bhospital\b", r"\bemergency medical\b",
        r"\bmedical treatment\b", r"\bmedical evacuation\b", r"\bmedical benefit\b",
    ],
    "baggage": [
        r"\bbaggage\b", r"\bluggage\b", r"\bpersonal effects\b",
        r"\bbaggage loss\b", r"\bbaggage delay\b", r"\bbaggage benefit\b",
    ],
}

_HANDBOOK_SECTION_PATTERNS: dict[str, list[str]] = {
    "definitions": [
        r"\bdefin", r"\bmeans?\b", r"\bshall mean\b", r"\brefers? to\b",
        r"\binterpretation\b", r"\bglossary\b",
    ],
    "principles": [
        r"\butmost good faith\b", r"\buberrima fide\b",
        r"\bsubrogation\b", r"\bcontribution\b",
        r"\binsurable interest\b", r"\bindemnity principle\b",
        r"\bproximate cause\b", r"\bprinciple of\b",
    ],
    "legislation": [
        r"\bact\b", r"\bsection \d", r"\bclause \d", r"\bschedule\b",
        r"\bregulation\b", r"\bnotification\b", r"\bgazette\b",
        r"\bprovision\b", r"\bamendment\b", r"\bstatute\b",
        r"\birda\b", r"\birdai\b",
    ],
    "case_law": [
        r"\bv\.\b", r"\bjudgment\b", r"\bjudgement\b",
        r"\bsupreme court\b", r"\bhigh court\b",
        r"\bheld that\b", r"\bcourt held\b",
        r"\bappeal\b", r"\bpetition\b", r"\bwrit\b",
    ],
    "history": [
        r"\bhistory\b", r"\bevolution\b", r"\borigin\b",
        r"\bnationaliz", r"\bnationalised\b",
        r"\bestablish", r"\bfounded\b", r"\bincorporated\b",
        r"\b19\d\d\b", r"\b20\d\d\b",
    ],
    "types_of_insurance": [
        r"\btypes of insurance\b", r"\bclassification\b",
        r"\bgeneral insurance\b", r"\blife insurance\b",
        r"\bmarine insurance\b", r"\bfire insurance\b",
        r"\bmotor insurance\b", r"\bhealth insurance\b",
        r"\bcrop insurance\b", r"\bmicro.?insurance\b",
    ],
    "chapter": [
        r"\bchapter\b", r"\bunit\b", r"\bmodule\b",
        r"\bintroduction\b", r"\boverview\b", r"\bbackground\b",
        r"\bsummary\b", r"\bconclusion\b",
    ],
}

# Backward-compat alias
_SECTION_PATTERNS = _POLICY_SECTION_PATTERNS


def _detect_section(text: str, doc_type: str = "policy_document") -> str:
    """
    Detect the most likely section label for a chunk of text using regex.

    Uses different pattern sets depending on document type:
      - policy_document              → policy-oriented labels (benefits, exclusions, …)
      - reference_handbook/regulatory → handbook labels (principles, case_law, …)
      - general/youtube/other         → falls back to policy patterns

    Requires ≥ 2 pattern hits to assign a section label — a single weak hit
    is not enough evidence and causes false-positive labels.
    """
    patterns = (
        _HANDBOOK_SECTION_PATTERNS
        if doc_type in ("reference_handbook", "regulatory")
        else _POLICY_SECTION_PATTERNS
    )
    t = text.lower()
    scores = {s: sum(1 for p in pats if re.search(p, t)) for s, pats in patterns.items()}
    best = max(scores, key=scores.__getitem__)
    return best if scores[best] >= 2 else "general"


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT ROUTING
# ══════════════════════════════════════════════════════════════════════════════
_DOCUMENT_ROUTING_MAP = [
    (["hajj", "umrah", "pilgrimage", "mecca", "rak travel", "outbound", "rak_travel"], "RAK_Travel_Outbound"),
    (["aig"], "AIG"),
    (["gig"], "GIG"),
    (["liva"], "LIVA"),
    (["rak"], "RAK"),
]

def _query_contains_term(query_lower: str, term: str) -> bool:
    q = re.sub(r"[_\-]", " ", query_lower)
    t = re.sub(r"[_\-]", " ", term.lower()).strip()
    if not t:
        return False
    if " " in t:
        return t in q
    return re.search(rf"\b{re.escape(t)}\b", q) is not None

def _route_to_documents(query: str, available_sources: list[str]) -> Optional[list[str]]:
    q = query.lower()
    matched_sources = []
    for keywords, tag in _DOCUMENT_ROUTING_MAP:
        if not any(_query_contains_term(q, kw) for kw in keywords):
            continue
        tag_lower = tag.lower().replace("_", " ").replace("-", " ")
        matched = [s for s in available_sources if tag_lower in s.lower().replace("_", " ").replace("-", " ")]
        for src in matched:
            if src not in matched_sources:
                matched_sources.append(src)
    if matched_sources:
        logger.info("[DOC ROUTER] Routed to %s", matched_sources)
        return matched_sources
    return None

# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL LOGIC DETECTOR
# ══════════════════════════════════════════════════════════════════════════════
_CONDITION_TRIGGERS = [
    r"\bonly if\b", r"\bunless\b", r"\bprovided that\b", r"\bsubject to\b",
    r"\bin the event\b", r"\bprovided\b", r"\bexcept\b", r"\bin case\b",
    r"\bif and only\b", r"\bcontingent\b", r"\bconditional\b",
]

def _extract_condition_hint(chunks: list[Document]) -> Optional[str]:
    conditions_found = []
    for chunk in chunks:
        text = chunk.page_content
        for pat in _CONDITION_TRIGGERS:
            for sent in re.split(r'[.\n]', text):
                if re.search(pat, sent, re.IGNORECASE) and len(sent.strip()) > 20:
                    conditions_found.append(sent.strip())
                    break
    if conditions_found:
        unique = list(dict.fromkeys(conditions_found))[:4]
        return "CONDITIONAL CLAUSES FOUND — handle with 'Covered only if …':\n" + "\n".join(f"  • {c}" for c in unique)
    return None

# ══════════════════════════════════════════════════════════════════════════════
# KEYWORD EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "to", "is", "are", "be",
    "for", "on", "at", "by", "with", "from", "this", "that", "which",
    "as", "it", "its", "not", "but", "if", "when", "where", "who",
    "will", "shall", "may", "can", "under", "above", "below", "per",
    "any", "all", "each", "such", "no", "yes", "has", "have", "had",
    "been", "was", "were", "does", "did", "do", "been", "being",
})

def _extract_keywords(text: str) -> list[str]:
    tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9\-]{2,}\b', text.lower())
    return list(dict.fromkeys(t for t in tokens if t not in _STOPWORDS))[:40]

# ══════════════════════════════════════════════════════════════════════════════
# QUERY CLEANING
# ══════════════════════════════════════════════════════════════════════════════
_STOP_WORDS = frozenset({
    # Question words
    "what", "who", "how", "when", "where", "why", "which",
    # Auxiliary verbs
    "is", "are", "was", "were", "be", "been", "being",
    "can", "could", "will", "would", "should", "shall",
    "may", "might", "must", "do", "does", "did", "done",
    "have", "has", "had", "get", "got",
    # Common verbs used in questions
    "tell", "explain", "describe", "define", "list", "show", "give",
    "mean", "means", "refer", "refers", "work", "works",
    # Articles, prepositions, conjunctions, pronouns
    "a", "an", "the",
    "in", "of", "on", "at", "by", "to", "for", "from", "with",
    "into", "onto", "upon", "about", "above", "below", "between",
    "and", "or", "but", "nor", "yet", "so",
    "if", "then", "than", "as", "that", "this", "these", "those",
    "me", "us", "my", "i", "you", "your", "we", "our",
    "it", "its", "they", "them", "their",
    # Filler phrases
    "please", "just", "really", "actually", "basically",
})


def _clean_query(query: str) -> str:
    """
    Strip punctuation and stop words from the query before retrieval.
    Keeps only the meaningful topic words so BM25 and dense search
    match document content rather than question structure.
    Example: 'What is Insurance?' → 'Insurance'
    The original query with punctuation is still used for LLM prompts.
    """
    # Remove punctuation
    cleaned = re.sub(r"[?!;:,\.]+", "", query).strip()
    # Filter stop words, keep tokens longer than 1 char
    tokens = [t for t in cleaned.split() if t.lower() not in _STOP_WORDS and len(t) > 1]
    result = " ".join(tokens)
    # Fall back to punctuation-stripped query if all tokens were stop words
    return result if result else cleaned


def _extract_query_intent(query: str) -> str:
    """
    Extract the core topic from a query by removing stop words.
    Example: 'What is a ULIP?' → 'ulip'
    """
    cleaned = re.sub(r"[?!.,;:]+", "", query.lower()).strip()
    tokens  = [t for t in cleaned.split() if t not in _STOP_WORDS and len(t) > 1]
    return " ".join(tokens) if tokens else cleaned

# ══════════════════════════════════════════════════════════════════════════════
# SECTION-AWARE CHUNKER
# ══════════════════════════════════════════════════════════════════════════════
class SectionChunker:
    """
    Wrapper around SemanticChunker that adds section/keyword metadata to chunks.
    All chunking (500-word chunks, 60-word overlap, page merging) is delegated
    to SemanticChunker internally.

    chunk_size / chunk_overlap accepted for backward compat but unused.
    """
    def __init__(
        self,
        chunk_size: int = 2000,      # accepted for backward compat, unused
        chunk_overlap: int = 600,    # accepted for backward compat, unused
        embed_model: Any = None,
        breakpoint_percentile: float = 90.0,
    ):
        self._splitter = SemanticChunker(
            embed_model=embed_model,
            breakpoint_pct=breakpoint_percentile,
        )

    def split_documents(
        self,
        docs: list[Document],
        doc_type: str = "policy_document",
        llm: Any = None,
    ) -> list[Document]:
        """
        Split documents and annotate each chunk with section/policy_type/keywords.
        Page merging and page-number recovery are handled by SemanticChunker.
        """
        chunks = self._splitter.split_documents(docs)

        for chunk in chunks:
            effective_doc_type = chunk.metadata.get("doc_type", doc_type)
            is_youtube = (
                effective_doc_type == "youtube"
                or "whisper" in str(chunk.metadata.get("source_type", "")).lower()
                or "youtube" in str(chunk.metadata.get("source", "")).lower()
            )

            if llm is not None:
                chunk.metadata["section"] = classify_chunk_intent(
                    chunk.page_content,
                    doc_type=effective_doc_type,
                    llm=llm,
                    force_llm=is_youtube,
                )
            else:
                chunk.metadata["section"] = _detect_section(
                    chunk.page_content, doc_type=effective_doc_type
                )

            chunk_policy = classify_chunk_policy_type(
                chunk.page_content,
                llm=llm if llm is not None else None,
                force_llm=is_youtube if llm is not None else False,
            )
            if chunk_policy != "general":
                chunk.metadata["policy_type"] = chunk_policy

            chunk.metadata["keywords"] = _extract_keywords(chunk.page_content)

        return chunks

# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def _chunks_within_context_limit(chunks: list[Document], max_chars: int) -> list[Document]:
    """Select complete chunks that fit the context budget."""
    selected = []
    used_chars = 0
    for chunk in chunks:
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "?")
        section = chunk.metadata.get("section", "general").title()
        rendered_length = len(
            f"[Section: {section} | Source: {source} | Page: {page}]\n{chunk.page_content}"
        )
        separator_length = 2 if selected else 0
        if selected and used_chars + separator_length + rendered_length > max_chars:
            continue
        selected.append(chunk)
        used_chars += separator_length + rendered_length
        if used_chars >= max_chars:
            break
    return selected


def _build_structured_context(chunks: list[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    _CHUNK_MIN  = 200   # guaranteed minimum chars per chunk
    _CHUNK_MAX  = 1500  # max chars per chunk (same for all source types)

    parts = []
    used = 0
    for chunk in chunks:
        remaining = max(max_chars - used, _CHUNK_MIN)
        allotted  = min(_CHUNK_MAX, remaining)

        section = chunk.metadata.get("section", "general").title()
        source  = chunk.metadata.get("source", "Unknown")
        page    = chunk.metadata.get("page", "?")
        header  = f"[Section: {section} | Source: {source} | Page: {page}]\n"

        body = chunk.page_content
        space_for_body = allotted - len(header)
        if space_for_body <= 0:
            body = body[:_CHUNK_MIN]
        elif len(body) > space_for_body:
            body = body[:space_for_body] + "…"

        part = header + body
        parts.append(part)
        used += len(part) + 2

        if used >= max_chars and chunk is not chunks[-1]:
            break

    return "\n\n".join(parts)

def _sources_from_chunks(chunks: list[Document]) -> list[str]:
    seen, result = set(), []
    for doc in chunks:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        key = (src, page)
        if key not in seen:
            seen.add(key)
            has_page = page not in (None, "", "?")
            result.append(f"{src} (page {page})" if has_page else src)
    return result

# ══════════════════════════════════════════════════════════════════════════════
# QUERY CLASSIFICATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
_ALL_DOCS_EXPLICIT = [
    "from all documents", "from all files", "from all resumes",
    "across all documents", "across all files", "across all resumes",
    "all documents", "all resumes", "all files",
    "from each document", "from each file", "from each resume",
    "each document", "each resume", "each file",
    "every document", "every resume", "every file",
    "extract from all", "extract all",
    "list all candidates", "list all resumes", "list all documents",
    "summary of all", "compare all",
]
_FIELD_MAP = [
    (["name", "candidate", "person", "insured", "policyholder", "holder"], "name"),
    (["email", "mail", "e-mail"], "email"),
    (["phone", "contact", "mobile"], "phone_number"),
    (["experience", "exp", "year"], "experience"),
    (["skill", "technology", "tech stack"], "skills"),
    (["education", "degree", "qualification"], "education"),
    (["company", "employer", "organisation", "organization", "worked at"], "current_company"),
    (["role", "designation", "position", "title", "job"], "designation"),
    (["policy number", "policy no", "policy id", "policy"], "policy_number"),
    (["covered", "coverage", "what is covered", "benefits", "benefit"], "coverage"),
    (["premium", "amount", "premium amount"], "premium"),
    (["sum insured", "sum assured", "coverage amount"], "sum_insured"),
    (["policy type", "plan type", "plan name"], "policy_type"),
    (["insurer", "insurance company", "provider"], "insurer"),
    (["expiry", "expiry date", "valid till", "end date"], "expiry_date"),
    (["start date", "commencement", "issue date", "inception"], "start_date"),
    (["nominee", "beneficiary"], "nominee"),
    (["exclusion", "not covered", "excluded"], "exclusions"),
    (["claim", "claim process", "claim procedure"], "claim_process"),
]
_COMPARISON_PHRASES = ["compare", "comparison", "vs", "versus", "difference between", "which is better", "which offers", "which insurer", "which policy", "all insurers", "all policies", "both", "each insurer", "across policies", "across insurers", "between"]
_PERSONAL_QUERY_WORDS = ["my flight", "my baggage", "my claim", "my policy", "my trip", "my luggage", "my travel", "my delay", " my ", "i was", "i am ", "i have", "i need", "i got", "i lost", "i missed", "i paid"]
_INFORMATIONAL_PHRASES = ["what is", "what are", "what does", "what do", "what's", "how much is", "how much does", "how much do", "how much can", "describe", "explain", "tell me about", "what coverage", "what benefit", "what limit", "what excess", "what deductible", "does it cover", "is there coverage", "is there a benefit", "list the", "show me the", "what type", "what kind", "under rak", "under aig", "under gig", "under liva"]
_SCENARIO_WORDS = ["hours delayed", "hour delay", "days delayed", "day delay", "missed my", "missed the", "lost my", "stolen", "trip cost", "total cost", "paid for", "booked", "i was delayed", "my flight was", "my baggage was", "calculate", "how much will", "how much would", "how much should", "how much can i", "how much do i"]

def _is_all_docs_query(question: str) -> bool:
    q = question.lower()
    return any(phrase in q for phrase in _ALL_DOCS_EXPLICIT)

def _is_comparison_query(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in _COMPARISON_PHRASES)

def _is_personal_query(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in _PERSONAL_QUERY_WORDS)

def _is_informational_query(question: str) -> bool:
    if _is_personal_query(question):
        return False
    q = question.lower()
    return any(p in q for p in _INFORMATIONAL_PHRASES)

def _is_scenario_query(question: str) -> bool:
    q = question.lower()
    return _is_personal_query(question) or any(w in q for w in _SCENARIO_WORDS)

def _fields_from_question(question: str) -> list[str]:
    q = question.lower().replace("_", " ")
    fields = []
    for keywords, field_name in _FIELD_MAP:
        if any(kw in q for kw in keywords):
            if field_name not in fields:
                fields.append(field_name)
    return fields or ["name", "policy_number", "coverage", "premium"]

# ══════════════════════════════════════════════════════════════════════════════
# HEALTH HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def wait_for_vllm(retries: int = 20, delay: int = 3) -> bool:
    for _ in range(retries):
        try:
            r = requests.get(f"{VLLM_HOST}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(delay)
    return False

def list_vllm_models() -> list[str]:
    try:
        r = requests.get(f"{VLLM_HOST}/v1/models", timeout=5)
        return [m["id"] for m in r.json().get("data", [])]
    except Exception:
        return []

# ══════════════════════════════════════════════════════════════════════════════
# RAG PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
class RAGPipeline:
    def __init__(self):
        self._vector_store = ChromaVectorStore()
        self._summary_store = SummaryStore()
        self._chunker = SectionChunker(
            embed_model=self._vector_store.embed_model,
        )
        self._compressor = ContextCompressor(
            embed_model=self._vector_store.embed_model,
            max_chars_per_chunk=LLM_CONTEXT_WINDOW_CHARS,
        )
        _cache_path = os.path.join(
            os.getenv("INSUREHUB_DATA_DIR", os.path.expanduser("~/.insurehub")),
            "cache",
            "query_kv_cache.json",
        )
        self._cache = QueryKVCache(
            cache_path=_cache_path,
            ttl_seconds=int(os.getenv("KV_CACHE_TTL", "3600")),
            max_entries=int(os.getenv("KV_CACHE_MAX_ENTRIES", "500")),
            sem_threshold=float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.78")),
        )
        logger.info("[RAGPipeline] KV cache ready — path=%s", _cache_path)

    @property
    def vector_store(self):
        return self._vector_store

    @property
    def summary_store(self):
        return self._summary_store

    @property
    def chunker(self):
        return self._chunker

    # ── Shared LLM accessor (lazy, temperature=0) ──────────────────────────
    def _get_llm(self):
        """Return a zero-temperature LLM for metadata classification tasks."""
        try:
            return get_insurance_llm(temperature=0)
        except Exception as exc:
            logger.warning("[RAG] Could not get LLM for metadata classification: %s", exc)
            return None

    # ── Summary-index maintenance (Stage 1 of two-stage retrieval) ─────────
    def _upsert_summary(
        self,
        chunks: list[Document],
        source: str,
        doc_meta: dict,
        llm: Any = None,
    ) -> None:
        """
        Generate a 200-300 word summary for a newly-ingested source and store
        it in the summary index, so knowledge_query() can use it for Stage 1
        retrieval. Never raises — a failure here should not fail ingestion,
        it just means that source is excluded from Stage-1 narrowing later
        (knowledge_query()'s guard treats un-summarized sources as always
        included, so this fails safe).
        """
        try:
            summary_llm = llm if llm is not None else self._get_llm()
            summary_text = generate_summary(chunks, source, doc_meta, summary_llm)
            self._summary_store.upsert(source, summary_text, doc_meta)
        except Exception as exc:
            logger.warning(
                "[RAG] Summary generation/upsert failed for source=%s: %s", source, exc
            )

    # ── Document ingestion ─────────────────────────────────────────────────
    def add_document(self, uploaded_file) -> int:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            raw_docs = load_document(tmp_path, uploaded_file.name)
            upload_id = uuid.uuid4().hex[:12]
            unique_source = f"{upload_id}_{uploaded_file.name}"

            preview = raw_docs[0].page_content[:5000] if raw_docs else ""
            extra_text = " ".join(d.page_content for d in raw_docs[1:4])[:5000]
            doc_type = classify_document_type(
                uploaded_file.name, preview, extra_text
            )
            logger.info("Document type for '%s': %s", uploaded_file.name, doc_type)

            llm = None
            try:
                llm = self._get_llm()
            except Exception:
                pass

            doc_tags = tag_document(
                uploaded_file.name, preview,
                extra_text=extra_text,
                doc_type=doc_type,
                llm=llm,
            )

            for raw_doc in raw_docs:
                raw_doc.metadata["doc_type"] = doc_type

            chunks = self._chunker.split_documents(
                raw_docs, doc_type=doc_type, llm=llm
            )

            _CHUNK_SKIP_FIELDS = {"insurer_confidence", "policy_type_confidence", "all_insurers", "all_policy_types"}
            for chunk in chunks:
                chunk.metadata["source"] = unique_source
                chunk.metadata["filename"] = uploaded_file.name
                chunk.metadata.update({k: v for k, v in doc_tags.items() if k not in _CHUNK_SKIP_FIELDS})
                chunk.metadata["doc_type"] = doc_type
                chunk.metadata.setdefault("source_type", "document")

            self._vector_store.add_documents(chunks)
            doc_meta = {"filename": uploaded_file.name, "doc_type": doc_type, **doc_tags}
            self._upsert_summary(chunks, unique_source, doc_meta, llm)
            return len(chunks)
        finally:
            os.unlink(tmp_path)

    def add_url(self, url: str) -> int:
        docs = load_url(url)
        if not docs:
            return 0

        upload_id = uuid.uuid4().hex[:12]
        unique_source = f"{upload_id}_{url[:40]}"

        is_youtube = any(
            "youtube" in str(doc.metadata.get("source_type", "")).lower() or
            "whisper" in str(doc.metadata.get("source_type", "")).lower() or
            "youtube.com" in url or "youtu.be" in url
            for doc in docs
        )
        source_type = "youtube" if is_youtube else "web"

        preview = docs[0].page_content[:5000] if docs else ""
        extra_text = " ".join(d.page_content for d in docs[1:4])[:5000]

        # For YouTube content always preserve doc_type="youtube" so the
        # semantic chunker knows to use word-window pseudo-sentences instead
        # of punctuation splitting (auto-generated captions have no punctuation).
        if is_youtube:
            doc_type = "youtube"
        else:
            doc_type = classify_document_type(url, preview, extra_text)
        logger.info("Document type for URL '%s': %s (youtube=%s)", url, doc_type, is_youtube)

        llm = None
        try:
            llm = self._get_llm()
        except Exception:
            pass

        doc_tags = tag_document(url, preview, extra_text=extra_text, doc_type=doc_type, llm=llm)

        for doc in docs:
            doc.metadata["doc_type"] = doc_type
            doc.metadata.setdefault("source_type", source_type)

        chunks = self._chunker.split_documents(docs, doc_type=doc_type, llm=llm)

        self._vector_store.delete_by_field("source_url", url)

        _CHUNK_SKIP_FIELDS = {"insurer_confidence", "policy_type_confidence", "all_insurers", "all_policy_types"}
        for chunk in chunks:
            chunk.metadata["source"] = unique_source
            chunk.metadata["source_url"] = url
            chunk.metadata["filename"] = url
            chunk.metadata.update({k: v for k, v in doc_tags.items() if k not in _CHUNK_SKIP_FIELDS})
            chunk.metadata["doc_type"] = doc_type
            chunk.metadata.setdefault("source_type", "youtube" if is_youtube else "document")

            chunk_policy = classify_chunk_policy_type(
                chunk.page_content,
                llm=llm,
                force_llm=is_youtube,
            )
            if chunk_policy != "general":
                chunk.metadata["policy_type"] = chunk_policy

        self._vector_store.add_documents(chunks)
        doc_meta = {"filename": url, "source_url": url, "doc_type": doc_type, **doc_tags}
        self._upsert_summary(chunks, unique_source, doc_meta, llm)
        return len(chunks)

    # ── Document management ────────────────────────────────────────────────
    def list_documents(self) -> list[str]:
        return self._vector_store.list_values("filename")

    def clear_documents(self) -> None:
        self._vector_store.delete_all()
        try:
            self._summary_store.delete_all()
        except Exception as exc:
            logger.warning("[RAG] Failed to clear summary store: %s", exc)

    def remove_document(self, doc_name: str) -> None:
        try:
            results = self._vector_store.collection.get(
                where={"filename": doc_name}, include=["metadatas"]
            )
            sources = {m.get("source") for m in (results.get("metadatas") or []) if m.get("source")}
        except Exception as exc:
            logger.warning("[RAG] Could not look up sources for '%s' before delete: %s", doc_name, exc)
            sources = set()
        self._vector_store.delete_by_field("filename", doc_name)
        for source in sources:
            try:
                self._summary_store.delete(source)
            except Exception as exc:
                logger.warning("[RAG] Failed to delete summary for source=%s: %s", source, exc)

    def get_document_tags(self, doc_name: str) -> dict:
        results = self._vector_store.collection.get(
            where={"filename": doc_name}, limit=1, include=["metadatas"]
        )
        if results.get("metadatas"):
            meta = results["metadatas"][0]
            return {
                "insurer": meta.get("insurer", "UNKNOWN"),
                "policy_type": meta.get("policy_type", "general"),
            }
        return {"insurer": "UNKNOWN", "policy_type": "general"}

    def get_full_content(self, source: str) -> str:
        results = self._vector_store.collection.get(
            where={"filename": source},
            include=["documents", "metadatas"],
        )
        documents = results.get("documents") or []
        metadatas = results.get("metadatas") or [{} for _ in documents]

        def page_key(item: tuple[str, dict]) -> tuple[int, object]:
            page = item[1].get("page")
            if isinstance(page, (int, float)):
                return (0, page)
            return (1, str(page or ""))

        ordered = sorted(zip(documents, metadatas), key=page_key)
        return "\n\n".join(document for document, _ in ordered)

    def summarize_url(self, url: str) -> tuple[str, list[str]]:
        full_text = self.get_full_content(url)
        if not full_text.strip():
            return "No content found for this URL.", []
        if len(full_text) > SUMMARY_MAX_CHARS:
            full_text = full_text[:SUMMARY_MAX_CHARS] + "... (truncated)"
        from prompt_template import URL_SUMMARY_PROMPT
        try:
            prompt = URL_SUMMARY_PROMPT.format(context=full_text, question="Summarize this content.")
        except Exception:
            prompt = (
                f"Please provide a comprehensive summary of the following web page content.\n"
                f"Include all key points, names, numbers, dates, and important details.\n\n"
                f"Content:\n{full_text}\n\nDetailed Summary:"
            )
        llm = get_insurance_llm(temperature=0.3)
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, [url]

    # ── Query entry point (backward compat) ────────────────────────────────
    def query(
        self,
        question: str,
        model: str,
        allowed_docs: Optional[list[str]] = None,
    ) -> tuple[str, list[str], Optional[pd.DataFrame]]:
        question = question.strip()
        if not question:
            return "Question cannot be empty.", [], None
        if _is_all_docs_query(question) and allowed_docs:
            return self._extract_all_docs(question, model, allowed_docs)
        answer, sources = self._rag_query(question, model, allowed_docs=allowed_docs)
        return answer, sources, None

    # ── Main knowledge-base Q&A pipeline (with HyDE and citation) ──────────
    def _expand_query(self, question: str) -> list[str]:
        """Generate query variations using HyDE."""
        hyde_prompt = ChatPromptTemplate.from_template(
            "Write a detailed hypothetical answer to the following question. "
            "Use insurance policy language. Do NOT use any real facts, just plausible text.\n\n"
            "Question: {question}\n\nHypothetical answer:"
        )
        llm = get_insurance_llm(temperature=0.5)
        chain = hyde_prompt | llm | StrOutputParser()
        executor = None
        try:
            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(chain.invoke, {"question": question})
            hypo = future.result(timeout=8)
            logger.debug("HyDE expansion succeeded (%d chars).", len(hypo))
            return [question, hypo[:500]]
        except concurrent.futures.TimeoutError:
            logger.warning("HyDE timed out after 8s - using original query only.")
            return [question]
        except Exception as exc:
            logger.warning("HyDE expansion failed: %s - using original query only.", exc)
            return [question]
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

    def _summary_stage1_filter(self, question: str) -> Optional[dict]:
        """
        Stage 1 of two-stage retrieval: narrow the Stage-2 chunk search to
        the sources whose summary best matches the question.

        Guard rail: a source that has no summary yet (ingested before this
        feature existed, or summary generation failed) is ALWAYS included
        regardless of the summary search result. That means this can only
        narrow the search among sources that already have a summary — it
        can never make an existing, un-summarized document unreachable.

        Returns None (meaning "no filter, search everything") whenever the
        summary index is empty, unavailable, or doesn't actually narrow
        anything down.
        """
        try:
            if self._summary_store.count() == 0:
                return None
            all_sources = set(self._vector_store.list_values("source"))
            if not all_sources:
                return None
            summarized = set(self._summary_store.list_sources())
            unsummarized = all_sources - summarized
            top_sources = set(self._summary_store.get_top_sources(question, top_k=SUMMARY_STAGE1_TOP_K))
            top_sources &= summarized  # drop any stale entries for already-deleted docs
            allowed = top_sources | unsummarized
            if not allowed or allowed >= all_sources:
                return None  # nothing to narrow
            return {"source": {"$in": sorted(allowed)}}
        except Exception as exc:
            logger.warning("[RAG] Summary-stage source narrowing failed (%s) — skipping it.", exc)
            return None

    def knowledge_query(self, question: str) -> tuple[str, bool, list[str]]:
        question = question.strip()
        if not question:
            return "Question cannot be empty.", False, []

        # Clean query for retrieval (strip ?, !, ;, : so BM25 tokens match).
        # LLM prompts and cache keys keep the original question with punctuation.
        search_question = _clean_query(question)
        logger.info(
            "[RAGPipeline] query=%r  search=%r  intent=%r",
            question[:80], search_question[:80], _extract_query_intent(question)[:60],
        )

        # ── KV cache lookup ────────────────────────────────────────────────
        _current_sources = self._vector_store.list_sources()
        _cache_key = QueryKVCache.make_key(
            query=question,
            top_k=RERANK_K,
            use_hybrid=True,
            use_reranker=True,
            generate_answer=True,
            run_ragas=False,
            sources=_current_sources,
        )

        # 1. Exact hit — same question, same source set, within TTL.
        _cached = self._cache.get(_cache_key)
        if _cached is not None:
            logger.info("[RAGPipeline] KV cache exact hit for query=%r", question[:80])
            return _cached["answer"], _cached["is_general"], _cached["sources"]

        # 2. Semantic hit — rephrased question close enough in embedding space.
        try:
            _q_emb = self._vector_store.embed_model.encode(
                [question], normalize_embeddings=True, show_progress_bar=False
            )[0]
            _sem_cached = self._cache.semantic_get(_q_emb)
            if _sem_cached is not None:
                logger.info("[RAGPipeline] KV cache semantic hit for query=%r", question[:80])
                return _sem_cached["answer"], _sem_cached["is_general"], _sem_cached["sources"]
        except Exception as _exc:
            logger.warning("[RAGPipeline] KV cache semantic lookup failed: %s", _exc)
            _q_emb = None
        else:
            pass  # _q_emb already set above

        urls = extract_urls(question)
        if urls:
            url = urls[0].rstrip(".,;:!?)]}")
            q_lower = question.lower()
            if any(p in q_lower for p in ["full text", "raw text"]):
                full_text = self.get_full_content(url)
                if not full_text.strip():
                    docs = load_url(url)
                    full_text = "\n\n".join(doc.page_content for doc in docs)
                return full_text or "No content found for this URL.", False, [url]
            docs = load_url(url)
            if docs:
                context = "\n\n".join(doc.page_content for doc in docs)
                answer = self._summarize_with_citations(context, question)
                return answer, False, [url]
            return "No content found for this URL.", False, [url]

        if self._vector_store.count() == 0:
            return "EMPTY_KB", True, []

        llm = None
        try:
            llm = self._get_llm()
        except Exception:
            pass
        query_meta = classify_query(question, llm=llm)
        routed_docs = _route_to_documents(question, self.list_documents())
        filter_meta = build_metadata_filter(query_meta, routed_docs)

        # Stage 1 (summaries) → Stage 2 (chunks): skipped when an explicit
        # keyword route already pinned the source set, since that's a more
        # specific signal than a semantic summary match.
        if not routed_docs:
            summary_filter = self._summary_stage1_filter(question)
            if summary_filter:
                filter_meta = (
                    {"$and": [filter_meta, summary_filter]} if filter_meta else summary_filter
                )

        expanded_queries = self._expand_query(search_question)

        # Retrieve with cleaned query for accurate BM25+dense matching.
        # Single reranker pass uses original question (cross-encoder reads natural language).
        all_chunks = []
        for q in expanded_queries:
            chunks = self._vector_store.search(
                q,
                top_k=RERANK_K,
                filter_metadata=filter_meta,
                use_hybrid=True,
                use_reranker=False,
            )
            all_chunks.extend(chunks)
        chunks = self._deduplicate_chunks(all_chunks)[:RETRIEVE_K]
        if len(chunks) > 1:
            chunks = self._vector_store.rerank_documents(question, chunks, top_k=RERANK_K)

        if not chunks:
            return "Not mentioned in documents.", False, []

        chunks = self._compressor.compress_to_budget(question, chunks, LLM_CONTEXT_WINDOW_CHARS)
        sources = _sources_from_chunks(chunks)

        context = _build_structured_context(chunks, max_chars=LLM_CONTEXT_WINDOW_CHARS)
        condition_hint = _extract_condition_hint(chunks)
        has_conflict, insurers = detect_conflict(chunks)
        conflict_hint = ""
        if has_conflict:
            conflict_hint = (
                f"The context contains multiple insurers "
                f"({', '.join(sorted(insurers))}). Keep each insurer's facts separate."
            )

        citation_prompt = f"""You are an Insurance Policy Analyst. Answer based ONLY on the CONTEXT below.

RULES (strictly enforced):
1. For every fact, number, condition, or limit, you MUST cite the exact source and page number like this: [Source: document_name, Page X].
2. If a piece of information is not present in the context, say "Not mentioned in documents."
3. Do not invent any information. If you are unsure, say "Not mentioned in documents."
4. Format your answer using markdown: headings, bullet points, bold for key numbers.
5. If the question asks for a calculation, show step‑by‑step using only numbers from context.
6. Do not combine limits or conditions from different insurers or policy documents.

RETRIEVAL NOTES:
{condition_hint or "No conditional-clause hint detected."}
{conflict_hint or "No multi-insurer conflict detected."}

CONTEXT:
{context}

QUESTION: {question}

ANSWER (with citations):"""

        llm = get_insurance_llm(temperature=0)
        response = llm.invoke(citation_prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        if "[Source:" not in answer and "Not mentioned" not in answer:
            answer += (
                "\n\n⚠️ **Warning:** The above answer could not be verified with explicit "
                "citations. Please verify against the original documents."
            )

        grounded, missing = validate_grounding(answer, context)
        if not grounded and missing:
            missing_values = ", ".join(sorted(str(m) for m in missing))
            answer += (
                f"\n\n⚠️ Warning: These figures could not be verified in the source documents: "
                f"{missing_values}. Please cross-check against the original policy document."
            )

        # ── Store result in KV cache ───────────────────────────────────────
        try:
            _payload = {"answer": answer, "is_general": False, "sources": sources}
            # Reuse the embedding computed at the top of this method if available,
            # otherwise skip the semantic index (exact-match cache still works).
            _store_emb = _q_emb if "_q_emb" in dir() and _q_emb is not None else None
            self._cache.put(_cache_key, _payload, query_embedding=_store_emb, query_text=question)
        except Exception as _exc:
            logger.warning("[RAGPipeline] KV cache put failed: %s", _exc)

        return answer, False, sources

    @staticmethod
    def _source_filter(sources: Optional[list[str]]) -> Optional[dict]:
        if not sources:
            return None
        unique_sources = list(dict.fromkeys(sources))
        if len(unique_sources) == 1:
            return {"source": unique_sources[0]}
        return {"source": {"$in": unique_sources}}

    @staticmethod
    def _content_fingerprint(text: str) -> str:
        normalised = re.sub(r"[^a-z0-9]", "", text.lower())
        return normalised[:200]

    @staticmethod
    def _deduplicate_chunks(chunks: list[Document]) -> list[Document]:
        """
        Remove duplicate and near-duplicate chunks from a retrieval result.

        Two-pass deduplication:
          Pass 1 (exact) — key = (source, page, full_text).
          Pass 2 (fingerprint) — key = content_fingerprint(text).

        Note: with overlap enabled, adjacent chunks share ~100 words of text.
        The fingerprint uses only the first 200 normalised chars, which is
        well within the non-overlapping body of each chunk, so overlapping
        chunks are NOT incorrectly deduplicated against each other.
        """
        def _score(chunk: Document) -> float:
            return chunk.metadata.get("rerank_score", chunk.metadata.get("similarity", 0.0))

        exact: dict[tuple[str, object, str], Document] = {}
        for chunk in chunks:
            key = (
                str(chunk.metadata.get("source", "Unknown")),
                chunk.metadata.get("page"),
                chunk.page_content,
            )
            existing = exact.get(key)
            if existing is None or _score(chunk) > _score(existing):
                exact[key] = chunk

        fingerprint: dict[str, Document] = {}
        for chunk in exact.values():
            fp = RAGPipeline._content_fingerprint(chunk.page_content)
            existing = fingerprint.get(fp)
            if existing is None or _score(chunk) > _score(existing):
                fingerprint[fp] = chunk

        logger.debug(
            "_deduplicate_chunks: %d in → %d after exact → %d after fingerprint",
            len(chunks), len(exact), len(fingerprint),
        )

        return sorted(fingerprint.values(), key=_score, reverse=True)

    def _summarize_with_citations(self, content: str, question: str) -> str:
        prompt = (
            f"Summarize the following web page content in a detailed, structured way. "
            f"Use headings, bullet points, and include all important facts (dates, numbers, names). "
            f"Do not add external knowledge.\n\nContent:\n{content[:6000]}\n\n"
            f"Question: {question}\n\nDetailed summary:"
        )
        llm = get_insurance_llm(temperature=0.3)
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def general_query(self, question: str) -> str:
        # Use temperature=0 to reduce hallucination; GENERAL_PROMPT now enforces
        # insurance-only scope and forbids answering from training knowledge.
        llm = get_general_llm(temperature=0)
        response = llm.invoke(GENERAL_PROMPT.format(question=question))
        return response.content if hasattr(response, "content") else str(response)

    def _rag_query(
        self,
        question: str,
        model: str,
        allowed_docs: Optional[list[str]] = None,
    ) -> tuple[str, list[str]]:
        llm = get_insurance_llm(temperature=0)
        filter_meta = self._source_filter(allowed_docs)
        chunks = self._vector_store.search(_clean_query(question), top_k=5, filter_metadata=filter_meta)
        if not chunks:
            return "Not mentioned in documents.", []
        chunks = self._compressor.compress_to_budget(question, chunks, LLM_CONTEXT_WINDOW_CHARS)
        context = _build_structured_context(chunks, max_chars=LLM_CONTEXT_WINDOW_CHARS)
        sources = _sources_from_chunks(chunks)
        response = llm.invoke(RAG_PROMPT.format(context=context, question=question))
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, sources

    # ── Bulk structured extraction ─────────────────────────────────────────
    def _extract_all_docs(
        self,
        question: str,
        model: str,
        doc_names: list[str],
    ) -> tuple[str, list[str], Optional[pd.DataFrame]]:
        llm = get_insurance_llm(temperature=0)
        fields = _fields_from_question(question)
        _FIELD_HINTS = {
            "name": "extract the full name of the insured person or policyholder.",
            "policy_number": 'look for "Policy No", "Policy Number", "Policy ID".',
            "coverage": 'look for "Sum Insured", "Coverage", "Benefits", "What is Covered".',
            "sum_insured": 'look for "Sum Insured", "Sum Assured", "Coverage Amount".',
            "insurer": "look for the insurance company name.",
            "policy_type": 'look for "Plan Name", "Policy Type", "Product Name".',
            "expiry_date": 'look for "Valid Till", "Expiry Date", "Policy End Date".',
            "start_date": 'look for "Inception Date", "Commencement Date".',
            "premium": 'look for "Premium Amount", "Annual Premium".',
            "nominee": 'look for "Nominee Name", "Beneficiary".',
            "exclusions": 'look for "Exclusions", "Not Covered".',
            "experience": "look for total years of work experience.",
            "skills": "look for technical skills, tools, programming languages.",
            "education": "look for degree, university, graduation year.",
            "current_company": "look for the most recent employer.",
            "designation": "look for current job title or most recent role.",
        }
        rows = []
        for doc_name in doc_names:
            results = self._vector_store.collection.get(
                where={"filename": doc_name},
                limit=50,
                include=["documents", "metadatas"],
            )
            documents = results.get("documents") or []
            metadatas = results.get("metadatas") or [{} for _ in documents]

            def page_key(item: tuple[str, dict]) -> tuple[int, object]:
                page = item[1].get("page")
                if isinstance(page, (int, float)):
                    return (0, page)
                return (1, str(page or ""))

            pairs = sorted(zip(documents, metadatas), key=page_key)
            raw_chunks = [document for document, _ in pairs]
            context = "\n\n".join(raw_chunks)[:6000]
            hints = "\n".join(
                f"- For {f}: {_FIELD_HINTS[f]}" for f in fields if f in _FIELD_HINTS
            )
            fields_str = ", ".join(f'"{f}"' for f in fields)
            prompt = (
                f"Extract data from this document. Reply with ONLY a single JSON object "
                f"using these EXACT keys: {fields_str}\n"
                f"Rules:\n- Use null if a field is not found.\n{hints}\n"
                f"- One value per field (string). If multiple, join with \", \".\n"
                f"- No explanation. No extra keys. Just the JSON.\n\n"
                f"Document ({doc_name}):\n{context}\n\nJSON:"
            )
            raw = llm.invoke(prompt)
            parsed = self._parse_json(raw.content if hasattr(raw, "content") else str(raw))
            for f in fields:
                parsed.setdefault(f, None)
            parsed = {f: parsed.get(f) for f in fields}
            parsed["file"] = doc_name
            rows.append(parsed)
        if not rows:
            return "No data extracted.", doc_names, None
        df = pd.DataFrame(rows)
        cols = ["file"] + [c for c in df.columns if c != "file"]
        df = df[cols]
        df.columns = [c.replace("_", " ").title() for c in df.columns]
        return f"Extracted from {len(rows)} document(s). Download Excel above.", doc_names, df

    def _find_doc_by_name_in_query(
        self,
        question: str,
        allowed_docs: Optional[list[str]] = None,
    ) -> Optional[str]:
        q_words = set(question.lower().split())
        doc_names = allowed_docs if allowed_docs else self.list_documents()
        best_doc, best_score = None, 0
        for doc_name in doc_names:
            stem = re.sub(r'[_\-.]', ' ', doc_name)
            stem = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)
            doc_words = set(
                w.lower() for w in stem.split()
                if w.lower() not in {"resume", "cv", "pdf", "updated", "doc"}
            )
            matches = len(q_words & doc_words)
            if matches > best_score:
                best_score = matches
                best_doc = doc_name
        return best_doc if best_score >= 2 else None

    @staticmethod
    def _parse_json(raw: str) -> dict:
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", raw):
            try:
                parsed, _ = decoder.raw_decode(raw[match.start():])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        return {}