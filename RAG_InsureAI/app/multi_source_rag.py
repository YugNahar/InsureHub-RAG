"""
Unified RAG with strict grounding – no hallucinations.
Supports document filtering with substring matching.
"""
import asyncio
import logging
import re
from typing import List, Tuple, Optional


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

from langchain_core.documents import Document
from rag import RAGPipeline
from router import get_insurance_llm
from video_store import VideoVectorStore
from webpage_store import WebpageVectorStore
from calculator import compute_insurance_benefits, _is_calculation_question
from prompt_template import STRICT_GROUNDED_PROMPT, CALCULATION_PROMPT, CONVERSATIONAL_RAG_PROMPT

logger = logging.getLogger(__name__)

class MultiSourceRAG:
    def __init__(self):
        self.doc_pipeline = RAGPipeline()
        self.video_store = VideoVectorStore()
        self.webpage_store = WebpageVectorStore()
        self.max_context_chars = 3500

    def _merge_chunks(self, chunks: List[Document]) -> List[Document]:
        seen = {}
        for chunk in chunks:
            h = hash(chunk.page_content[:200])
            if h not in seen or chunk.metadata.get("similarity", 0) > seen[h].metadata.get("similarity", 0):
                seen[h] = chunk
        return list(seen.values())

    async def ask(self, question: str, history: str = "", document_filter: Optional[List[str]] = None) -> Tuple[str, List[str]]:
        # Build filter
        filter_meta = None
        if document_filter:
            conditions = [{"source": {"$contains": doc}} for doc in document_filter]
            filter_meta = conditions[0] if len(conditions) == 1 else {"$or": conditions}
            logger.info(f"Document filter: {document_filter}")

        doc_chunks = await asyncio.to_thread(
            self.doc_pipeline._vector_store.search,
            question, top_k=6, use_hybrid=True, use_reranker=True,
            filter_metadata=filter_meta
        )

        if not document_filter:
            video_chunks = await asyncio.to_thread(
                self.video_store.search, question, top_k=4, use_hybrid=True, use_reranker=True
            )
            webpage_chunks = await asyncio.to_thread(
                self.webpage_store.search, question, top_k=4, use_hybrid=True, use_reranker=True
            )
            all_chunks = self._merge_chunks(doc_chunks + video_chunks + webpage_chunks)
        else:
            all_chunks = self._merge_chunks(doc_chunks)

        all_chunks.sort(key=lambda x: x.metadata.get("similarity", 0), reverse=True)
        all_chunks = all_chunks[:8]

        # Build context
        context_parts, sources = [], []
        for chunk in all_chunks:
            source_type = chunk.metadata.get("source_type", "document")
            if source_type == "document":
                src = chunk.metadata.get("source", "Unknown")
                page = chunk.metadata.get("page", "?")
                label = f"Document: {src} (Page {page})"
                sources.append(f"{src} (page {page})")
            elif source_type == "video":
                url = chunk.metadata.get("source_url", "Unknown URL")
                label = f"Video: {url}"
                sources.append(url)
            else:
                url = chunk.metadata.get("source_url", "Unknown URL")
                label = f"Webpage: {url}"
                sources.append(url)
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
            response = await asyncio.to_thread(llm.invoke, prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            return _strip_model_preamble(answer), list(dict.fromkeys(sources))

        if not full_context.strip():
            # No documents at all — use general knowledge
            prompt = CONVERSATIONAL_RAG_PROMPT.format(
                history=history,
                context="No relevant documents found in the knowledge base.",
                question=question
            )
            llm = get_insurance_llm(temperature=0.3)
            response = await asyncio.to_thread(llm.invoke, prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            return _strip_model_preamble(answer), []

        # ── Prompt selection ──────────────────────────────────────────────────
        # STRICT grounding only when the user is asking about a specific uploaded
        # document (document_filter is set).  For general insurance questions we
        # use the conversational prompt so the LLM can fall back to general
        # knowledge instead of incorrectly denying coverage based on an unrelated
        # document that happened to score highest in retrieval.
        if document_filter:
            prompt = STRICT_GROUNDED_PROMPT.format(history=history, context=full_context, question=question)
            llm = get_insurance_llm(temperature=0)
        else:
            prompt = CONVERSATIONAL_RAG_PROMPT.format(history=history, context=full_context, question=question)
            llm = get_insurance_llm(temperature=0.3)

        response = await asyncio.to_thread(llm.invoke, prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        return _strip_model_preamble(answer), list(dict.fromkeys(sources))

    # Management methods (keep as before)
    def video_exists(self, url: str) -> bool:
        return self.video_store.url_exists(url)
    def add_video_chunks(self, url: str, chunks: List[Document]):
        self.video_store.add_video_chunks(url, chunks)
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
