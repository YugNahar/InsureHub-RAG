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
from typing import Optional

import pandas as pd
import requests
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from document_loader import load_document, load_url, extract_urls
from metadata_tagger import tag_document
from validator import detect_conflict, validate_grounding
from router import get_insurance_llm, get_general_llm, VLLM_HOST
from prompt_template import (
    GENERAL_PROMPT,
    RAG_PROMPT,
)
from vector_store import ChromaVectorStore

logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────
RETRIEVE_K = 12
RERANK_K = 6
MAX_CONTEXT_CHARS = 6000
SUMMARY_MAX_CHARS = 20000

# ══════════════════════════════════════════════════════════════════════════════
# SECTION DETECTION (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
_SECTION_PATTERNS: dict[str, list[str]] = {
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

def _detect_section(text: str) -> str:
    t = text.lower()
    scores = {s: sum(1 for p in pats if re.search(p, t)) for s, pats in _SECTION_PATTERNS.items()}
    best = max(scores, key=scores.__getitem__)
    return best if scores[best] > 0 else "general"

# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT ROUTING (unchanged)
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
# CONDITIONAL LOGIC DETECTOR (unchanged)
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
# KEYWORD EXTRACTION (unchanged)
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
# SECTION-AWARE CHUNKER (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class SectionChunker:
    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 120):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=False,
        )
    def split_documents(self, docs: list[Document]) -> list[Document]:
        chunks = []
        for doc in docs:
            raw = self._splitter.split_documents([doc])
            for chunk in raw:
                chunk.metadata["section"] = _detect_section(chunk.page_content)
                chunk.metadata["keywords"] = _extract_keywords(chunk.page_content)
            chunks.extend(raw)
        return chunks

# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER (unchanged)
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
# QUERY CLASSIFICATION HELPERS (unchanged)
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
# HEALTH HELPERS (unchanged)
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
# RAG PIPELINE — TurboVec Vector Edition with HyDE, Hybrid Search, Citation
# ══════════════════════════════════════════════════════════════════════════════
class RAGPipeline:
    def __init__(self):
        self._vector_store = ChromaVectorStore()
        self._chunker = SectionChunker(chunk_size=600, chunk_overlap=80)

    @property
    def vector_store(self):
        """Public accessor for the underlying vector store."""
        return self._vector_store

    @property
    def chunker(self):
        """Public accessor for the document chunker."""
        return self._chunker

    # ── Document ingestion ─────────────────────────────────────────────────
    def add_document(self, uploaded_file) -> int:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        try:
            raw_docs = load_document(tmp_path, uploaded_file.name)
            chunks = self._chunker.split_documents(raw_docs)
            upload_id = uuid.uuid4().hex[:12]
            unique_source = f"{upload_id}_{uploaded_file.name}"
            preview = raw_docs[0].page_content[:600] if raw_docs else ""
            doc_tags = tag_document(uploaded_file.name, preview)
            for chunk in chunks:
                chunk.metadata["source"] = unique_source
                chunk.metadata["filename"] = uploaded_file.name
                chunk.metadata.update(doc_tags)
            self._vector_store.add_documents(chunks)
            return len(chunks)
        finally:
            os.unlink(tmp_path)

    def add_url(self, url: str) -> int:
        docs = load_url(url)
        chunks = self._chunker.split_documents(docs)
        upload_id = uuid.uuid4().hex[:12]
        unique_source = f"{upload_id}_{url[:40]}"
        self._vector_store.delete_by_field("source_url", url)
        for chunk in chunks:
            chunk.metadata["source"] = unique_source
            chunk.metadata["source_url"] = url
        self._vector_store.add_documents(chunks)
        return len(chunks)

    # ── Document management ─────────────────────────────────────────────────
    def list_documents(self) -> list[str]:
        return self._vector_store.list_values("filename")

    def clear_documents(self) -> None:
        self._vector_store.delete_all()

    def remove_document(self, doc_name: str) -> None:
        self._vector_store.delete_by_field("filename", doc_name)

    def get_document_tags(self, doc_name: str) -> dict:
        results = self._vector_store.collection.get(where={"filename": doc_name}, limit=1, include=["metadatas"])
        if results.get("metadatas"):
            meta = results["metadatas"][0]
            return {"insurer": meta.get("insurer", "UNKNOWN"), "policy_type": meta.get("policy_type", "general")}
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
            prompt = f"Please provide a comprehensive summary of the following web page content.\nInclude all key points, names, numbers, dates, and important details.\n\nContent:\n{full_text}\n\nDetailed Summary:"
        llm = get_insurance_llm(temperature=0.3)
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, [url]

    # ── Query entry point (backward compat) ────────────────────────────────
    def query(self, question: str, model: str, allowed_docs: Optional[list[str]] = None) -> tuple[str, list[str], Optional[pd.DataFrame]]:
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
            "Use insurance policy language. Do NOT use any real facts, just plausible text.\n\nQuestion: {question}\n\nHypothetical answer:"
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

        # URL questions do not depend on the document knowledge base.
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

        # ── Query expansion (HyDE) ─────────────────────────────────────────
        routed_docs = _route_to_documents(question, self.list_documents())
        filter_meta = self._source_filter(routed_docs)
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

        # ── Build context with forced citations ────────────────────────────
        context = _build_structured_context(chunks, max_chars=MAX_CONTEXT_CHARS)
        condition_hint = _extract_condition_hint(chunks)
        has_conflict, insurers = detect_conflict(chunks)
        conflict_hint = ""
        if has_conflict:
            conflict_hint = (
                "The context contains multiple insurers "
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

        # Post‑processing: warn if no citations found
        if "[Source:" not in answer and "Not mentioned" not in answer:
            answer += "\n\n⚠️ **Warning:** The above answer could not be verified with explicit citations. Please verify against the original documents."

        # Grounding validation
        grounded, missing = validate_grounding(answer, context)
        if not grounded and missing:
            missing_values = ", ".join(sorted(str(m) for m in missing))
            answer += f"\n\n⚠️ Warning: These figures could not be verified in the source documents: {missing_values}. Please cross-check against the original policy document."

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
    def _deduplicate_chunks(chunks: list[Document]) -> list[Document]:
        unique: dict[tuple[str, object, str], Document] = {}
        for chunk in chunks:
            key = (
                str(chunk.metadata.get("source", "Unknown")),
                chunk.metadata.get("page"),
                chunk.page_content,
            )
            existing = unique.get(key)
            chunk_score = chunk.metadata.get("rerank_score", chunk.metadata.get("similarity", 0))
            existing_score = (
                existing.metadata.get("rerank_score", existing.metadata.get("similarity", 0))
                if existing
                else float("-inf")
            )
            if existing is None or chunk_score > existing_score:
                unique[key] = chunk
        return sorted(
            unique.values(),
            key=lambda chunk: chunk.metadata.get(
                "rerank_score",
                chunk.metadata.get("similarity", 0),
            ),
            reverse=True,
        )

    def _summarize_with_citations(self, content: str, question: str) -> str:
        prompt = f"""Summarize the following web page content in a detailed, structured way (like Perplexity). Use headings, bullet points, and include all important facts (dates, numbers, names). Do not add external knowledge.

Content:
{content[:6000]}

Question: {question}

Detailed summary:"""
        llm = get_insurance_llm(temperature=0.3)
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    # ── URL / general queries ──────────────────────────────────────────────
    def general_query(self, question: str) -> str:
        llm = get_general_llm(temperature=0.7)
        response = llm.invoke(GENERAL_PROMPT.format(question=question))
        return response.content if hasattr(response, "content") else str(response)

    def _rag_query(self, question: str, model: str, allowed_docs: Optional[list[str]] = None) -> tuple[str, list[str]]:
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
    def _extract_all_docs(self, question: str, model: str, doc_names: list[str]) -> tuple[str, list[str], Optional[pd.DataFrame]]:
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
            hints = "\n".join(f"- For {f}: {_FIELD_HINTS[f]}" for f in fields if f in _FIELD_HINTS)
            fields_str = ", ".join(f'"{f}"' for f in fields)
            prompt = f"Extract data from this document. Reply with ONLY a single JSON object using these EXACT keys: {fields_str}\nRules:\n- Use null if a field is not found.\n{hints}\n- One value per field (string). If multiple, join with \", \".\n- No explanation. No extra keys. Just the JSON.\n\nDocument ({doc_name}):\n{context}\n\nJSON:"
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

    def _find_doc_by_name_in_query(self, question: str, allowed_docs: Optional[list[str]] = None) -> Optional[str]:
        q_words = set(question.lower().split())
        doc_names = allowed_docs if allowed_docs else self.list_documents()
        best_doc, best_score = None, 0
        for doc_name in doc_names:
            stem = re.sub(r'[_\-.]', ' ', doc_name)
            stem = re.sub(r'([a-z])([A-Z])', r'\1 \2', stem)
            doc_words = set(w.lower() for w in stem.split() if w.lower() not in {"resume", "cv", "pdf", "updated", "doc"})
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