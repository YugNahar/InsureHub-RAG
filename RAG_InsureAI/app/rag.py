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

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
RETRIEVE_K = 16
RERANK_K = 8
# 3500 chars ≈ 875 tokens of context.  Total input ≈ 1100 tokens (prompt+context+question)
# → prefill is ~1 second; 200 max_tokens of output ≈ 8 seconds at 24 tok/s on vLLM.
# Context compressor already strips irrelevant sentences, so 3500 chars is still rich.
MAX_CONTEXT_CHARS = 3500
SUMMARY_MAX_CHARS = 20000

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
# SECTION-AWARE CHUNKER
# ══════════════════════════════════════════════════════════════════════════════
class SectionChunker:
    """
    Args
    ----
    chunk_size, chunk_overlap : kept for backward-compatible call signatures
        (e.g. SectionChunker(chunk_size=600, chunk_overlap=80)). chunk_size
        is now used as the semantic chunker's max_chunk_chars ceiling and
        chunk_overlap as a floor for min_chunk_chars — chunk_overlap no
        longer produces literal overlapping text, since semantic boundaries
        are chosen at low-similarity points and don't need it.
    embed_model : the shared embedding model (e.g. ChromaVectorStore's
        embed_model). Pass this in to avoid loading a second copy of the
        model — see semantic_chunker.py.
    breakpoint_percentile : forwarded to SemanticChunker; higher = fewer,
        larger chunks.
    """
    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        embed_model: Any = None,
        breakpoint_percentile: float = 90.0,
    ):
        self._splitter = SemanticChunker(
            embed_model=embed_model,
            breakpoint_percentile=breakpoint_percentile,
            min_chunk_chars=max(150, chunk_overlap),
            max_chunk_chars=chunk_size,
        )

    def split_documents(
        self,
        docs: list[Document],
        doc_type: str = "policy_document",
        llm: Any = None,
    ) -> list[Document]:
        """
        Split documents into overlapping chunks and annotate each chunk with:
          - section: intent/section label (regex fast-path; LLM for ambiguous
            or YouTube/conversational chunks when llm is provided)
          - policy_type: per-chunk policy type (regex fast-path; LLM for
            ambiguous or YouTube chunks when llm is provided)
          - keywords: extracted keyword list

        Args:
            docs:     List of LangChain Document objects to split.
            doc_type: Document type from classify_document_type().
                      Controls which section-label vocabulary is used.
                      Defaults to "policy_document" for backward compatibility.
            llm:      Optional LangChain LLM instance. When provided, used for
                      chunk-level intent and policy_type classification on
                      ambiguous or YouTube/conversational chunks.
        """
        chunks = []
        for doc in docs:
            # Inherit doc_type from document metadata if present
            effective_doc_type = doc.metadata.get("doc_type", doc_type)

            # Determine if this is YouTube/conversational content
            is_youtube = effective_doc_type in ("youtube",) or \
                         "whisper" in str(doc.metadata.get("source_type", "")) or \
                         "youtube" in str(doc.metadata.get("source", "")).lower()

            raw = self._splitter.split_documents([doc])
            for chunk in raw:
                # ── Section / intent label ─────────────────────────────────
                # For YouTube chunks, always use LLM (regex misses colloquial).
                # For policy/handbook chunks, regex fast-path first; LLM only
                # when regex is ambiguous (handled inside classify_chunk_intent).
                if llm is not None:
                    chunk.metadata["section"] = classify_chunk_intent(
                        chunk.page_content,
                        doc_type=effective_doc_type,
                        llm=llm,
                        force_llm=is_youtube,
                    )
                else:
                    # No LLM: use regex-only _detect_section (backward compat)
                    chunk.metadata["section"] = _detect_section(
                        chunk.page_content, doc_type=effective_doc_type
                    )

                # ── Per-chunk policy_type ──────────────────────────────────
                # Always classify at chunk level — overrides the doc-level
                # "general" that tag_document() assigns to non-policy documents.
                # This means a handbook chapter on "motor claims" correctly
                # gets policy_type="motor" rather than "general".
                if llm is not None:
                    chunk_policy = classify_chunk_policy_type(
                        chunk.page_content,
                        llm=llm,
                        force_llm=is_youtube,
                    )
                else:
                    chunk_policy = classify_chunk_policy_type(
                        chunk.page_content,
                        llm=None,
                    )

                # Only override the doc-level policy_type if we found something
                # more specific than "general" at the chunk level.
                if chunk_policy != "general":
                    chunk.metadata["policy_type"] = chunk_policy
                # else: keep whatever the doc-level tag set (could be a real
                # policy type for policy_document, or "general" for others)

                chunk.metadata["keywords"] = _extract_keywords(chunk.page_content)
            chunks.extend(raw)
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
    parts = []
    for chunk in chunks:
        section = chunk.metadata.get("section", "general").title()
        source = chunk.metadata.get("source", "Unknown")
        page = chunk.metadata.get("page", "?")
        parts.append(f"[Section: {section} | Source: {source} | Page: {page}]\n{chunk.page_content}")
    full = "\n\n".join(parts)
    if len(full) > max_chars:
        full = full[:max_chars] + "... (truncated)"
    return full

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
        # Pass the embed_model already loaded by ChromaVectorStore/TurboVec
        # so the chunker doesn't load a second copy of the model.
        self._chunker = SectionChunker(
            chunk_size=600, chunk_overlap=80, embed_model=self._vector_store.embed_model
        )

    @property
    def vector_store(self):
        return self._vector_store

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

            # ── Step 1: Classify document type ─────────────────────────────
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

            # ── Step 2: Tag document (skips keyword matching for non-policy)
            doc_tags = tag_document(
                uploaded_file.name, preview,
                extra_text=extra_text,
                doc_type=doc_type,
                llm=llm,
            )

            # ── Step 3: Annotate raw docs so chunker inherits doc_type ──────
            for raw_doc in raw_docs:
                raw_doc.metadata["doc_type"] = doc_type

            # ── Step 4: Get LLM for per-chunk classification ─────────────

            # ── Step 5: Chunk with doc-type-aware section + policy_type ────
            # Pass llm so chunker can call classify_chunk_intent() and
            # classify_chunk_policy_type() on each chunk.
            chunks = self._chunker.split_documents(
                raw_docs, doc_type=doc_type, llm=llm
            )

            # ── Step 6: Attach source + doc-level tags to every chunk ───────
            for chunk in chunks:
                chunk.metadata["source"] = unique_source
                chunk.metadata["filename"] = uploaded_file.name
                # Apply doc-level tags first (insurer, policy_type, etc.)
                chunk.metadata.update(doc_tags)
                # doc_type must always be the classified value — update() above
                # may overwrite it from doc_tags (which also sets doc_type),
                # but we re-assert it explicitly to be safe.
                chunk.metadata["doc_type"] = doc_type
                # Restore the chunk-level policy_type if chunker set a more
                # specific one (chunk.metadata["policy_type"] was set by
                # split_documents BEFORE update() overwrote it with doc_tags).
                # We need to re-run per-chunk policy_type after the update.
                # Fix: classify_chunk_policy_type is idempotent — re-classify
                # using the already-embedded text (cheap since regex is fast).
                chunk_policy = classify_chunk_policy_type(
                    chunk.page_content, llm=llm,
                    force_llm=False,  # regex already ran; LLM only if ambiguous
                )
                if chunk_policy != "general":
                    chunk.metadata["policy_type"] = chunk_policy
                # If chunk_policy == "general", keep the doc-level policy_type
                # (from doc_tags) which may be more specific for policy docs.

            self._vector_store.add_documents(chunks)
            return len(chunks)
        finally:
            os.unlink(tmp_path)

    def add_url(self, url: str) -> int:
        """
        Load content from a URL (web page or YouTube), classify and tag it,
        then ingest into the vector store with full chunk-level metadata.

        Fixed vs original:
        - Classifies doc_type (was missing — all URL chunks had no doc_type).
        - Calls tag_document() for doc-level insurer/policy_type.
        - Calls classify_chunk_intent() + classify_chunk_policy_type() per chunk
          so even YouTube/conversational content gets meaningful section and
          policy_type metadata rather than always "general".
        - Detects YouTube URLs and sets force_llm=True for those chunks.
        """
        docs = load_url(url)
        if not docs:
            return 0

        upload_id = uuid.uuid4().hex[:12]
        unique_source = f"{upload_id}_{url[:40]}"

        # ── Detect source type for YouTube-aware classification ──────────
        is_youtube = any(
            "youtube" in str(doc.metadata.get("source_type", "")).lower() or
            "whisper" in str(doc.metadata.get("source_type", "")).lower() or
            "youtube.com" in url or "youtu.be" in url
            for doc in docs
        )
        source_type = "youtube" if is_youtube else "web"

        # ── Classify document type ────────────────────────────────────────
        preview = docs[0].page_content[:5000] if docs else ""
        extra_text = " ".join(d.page_content for d in docs[1:4])[:5000]
        # YouTube content is always "general" at the doc level — chunk-level
        # classification will provide the real signal.
        doc_type = classify_document_type(url, preview, extra_text)
        logger.info("Document type for URL '%s': %s (youtube=%s)", url, doc_type, is_youtube)

        llm = None
        try:
            llm = self._get_llm()
        except Exception:
            pass

        # ── Doc-level tag ─────────────────────────────────────────────────
        doc_tags = tag_document(url, preview, extra_text=extra_text, doc_type=doc_type, llm=llm)

        # ── Annotate raw docs with doc_type and source_type ───────────────
        for doc in docs:
            doc.metadata["doc_type"] = doc_type
            doc.metadata.setdefault("source_type", source_type)

        # ── Chunk with per-chunk intent + policy_type classification ──────
        chunks = self._chunker.split_documents(docs, doc_type=doc_type, llm=llm)

        # ── Delete any previous version of this URL then ingest ───────────
        self._vector_store.delete_by_field("source_url", url)

        for chunk in chunks:
            chunk.metadata["source"] = unique_source
            chunk.metadata["source_url"] = url
            chunk.metadata["filename"] = url
            chunk.metadata.update(doc_tags)
            chunk.metadata["doc_type"] = doc_type

            # Re-classify per-chunk policy_type after update() (same fix as
            # add_document — update() overwrites the chunker's chunk-level
            # policy_type with the doc-level one from doc_tags).
            chunk_policy = classify_chunk_policy_type(
                chunk.page_content,
                llm=llm,
                force_llm=is_youtube,   # always use LLM for YouTube
            )
            if chunk_policy != "general":
                chunk.metadata["policy_type"] = chunk_policy

        self._vector_store.add_documents(chunks)
        return len(chunks)

    # ── Document management ────────────────────────────────────────────────
    def list_documents(self) -> list[str]:
        return self._vector_store.list_values("filename")

    def clear_documents(self) -> None:
        self._vector_store.delete_all()

    def remove_document(self, doc_name: str) -> None:
        self._vector_store.delete_by_field("filename", doc_name)

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

    def knowledge_query(self, question: str) -> tuple[str, bool, list[str]]:
        question = question.strip()
        if not question:
            return "Question cannot be empty.", False, []

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

        # ── Query expansion (HyDE) + metadata filter ──────────────────────
        llm = None
        try:
            llm = self._get_llm()
        except Exception:
            pass
        query_meta = classify_query(question, llm=llm)
        routed_docs = _route_to_documents(question, self.list_documents())
        filter_meta = build_metadata_filter(query_meta, routed_docs)
        expanded_queries = self._expand_query(question)

        all_chunks = []
        for q in expanded_queries:
            chunks = self._vector_store.search(
                q,
                top_k=RERANK_K,
                filter_metadata=filter_meta,
                use_hybrid=True,
                use_reranker=True,
            )
            all_chunks.extend(chunks)
        chunks = self._deduplicate_chunks(all_chunks)[:RETRIEVE_K]

        if not chunks:
            return "Not mentioned in documents.", False, []

        chunks = _chunks_within_context_limit(chunks, MAX_CONTEXT_CHARS)
        sources = _sources_from_chunks(chunks)

        context = _build_structured_context(chunks, max_chars=MAX_CONTEXT_CHARS)
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
        llm = get_general_llm(temperature=0.7)
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
        chunks = self._vector_store.search(question, top_k=5, filter_metadata=filter_meta)
        if not chunks:
            return "Not mentioned in documents.", []
        chunks = _chunks_within_context_limit(chunks, MAX_CONTEXT_CHARS)
        context = _build_structured_context(chunks, max_chars=MAX_CONTEXT_CHARS)
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