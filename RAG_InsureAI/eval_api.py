"""
RAG Evaluation API — eval_api.py  (testing version)

KEY PERFORMANCE FIX vs previous version
-----------------------------------------
Root cause of 5-minute ingestion: SectionChunker.split_documents() was calling
classify_chunk_intent() + classify_chunk_policy_type() once per chunk with
llm=llm, and then _classify_docs_metadata() re-ran the same two LLM calls on
every chunk as a "safety net" — doubling the total LLM calls. At ~2-3s per
serial vLLM call, a 200-chunk document triggered 400-800 sequential LLM calls.

Fixes applied:
  1. SectionChunker called with llm=None — uses fast regex-only classification
     during chunking (zero LLM calls, ~ms total).
  2. _classify_docs_metadata() runs ONE pass after chunking, but:
       - Only calls LLM when regex is NOT confident (force_llm=False for docs).
       - For YouTube, force_llm=True but runs in a ThreadPoolExecutor so all
         YouTube chunks are classified in parallel, not serially.
       - Trims chunk text to 150 chars before the LLM call — enough signal,
         much shorter prompt.
  3. compress_to_budget fires only when total context > LLM_CONTEXT_WINDOW_CHARS,
     matching the fix applied to rag.py and multi_source_rag.py.
  4. Singleton LLM — created once on startup, never recreated per call.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.join(_here, "app"))

from app.document_loader import (
    load_document, is_youtube_url, _load_youtube,
    load_url_advanced,
)
from app.metadata_tagger import (
    tag_document, classify_query, build_metadata_filter, classify_document_type,
    classify_chunk_intent, classify_chunk_intent_batch,
    classify_chunk_policy_type,
)
from app.vector_store import ChromaVectorStore
from app.rag import (
    RAGPipeline, _detect_section, _build_structured_context,
    _sources_from_chunks, _extract_condition_hint,
    _route_to_documents,
    _clean_query, _extract_query_intent,
    MAX_CONTEXT_CHARS, RETRIEVE_K, RERANK_K, SUMMARY_STAGE1_TOP_K,
    LLM_CONTEXT_WINDOW_CHARS,
)
from app.validator import detect_conflict, validate_grounding
from app.context_compressor import ContextCompressor
from app.router import (
    get_insurance_llm,
    get_active_model_info,
    list_vllm_models,
    set_model_override,
)
from app.summary_store import SummaryStore
from app.summarizer import generate_summary
from app.kv_cache import QueryKVCache

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

_SAFE_TMP_ROOT = os.path.expanduser("~/.insurehub_tmp")
os.makedirs(_SAFE_TMP_ROOT, exist_ok=True)

# ── Shared singletons ──────────────────────────────────────────────────────────
_vector_store:  Optional[ChromaVectorStore] = None
_rag_pipeline:  Optional[RAGPipeline]       = None
_summary_store: Optional[SummaryStore]      = None
_kv_cache:      Optional[QueryKVCache]      = None
_compressor:    Optional[ContextCompressor] = None
_llm_instance:  Optional[Any]               = None

_KV_CACHE_PATH = os.path.join(_here, "app", "turbovec_data", "kv_cache.json")


def get_vector_store() -> ChromaVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore()
    return _vector_store

def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline

def get_summary_store() -> SummaryStore:
    global _summary_store
    if _summary_store is None:
        _summary_store = SummaryStore()
    return _summary_store

def get_kv_cache() -> QueryKVCache:
    global _kv_cache
    if _kv_cache is None:
        ttl = int(os.getenv("KV_CACHE_TTL", "3600"))
        _kv_cache = QueryKVCache(cache_path=_KV_CACHE_PATH, ttl_seconds=ttl)
    return _kv_cache

def get_compressor() -> ContextCompressor:
    global _compressor
    if _compressor is None:
        _compressor = ContextCompressor(
            embed_model=get_vector_store().embed_model,
            similarity_threshold=float(os.getenv("COMPRESS_THRESHOLD", "0.38")),
            min_sentences=int(os.getenv("COMPRESS_MIN_SENTS", "2")),
            max_sentences=int(os.getenv("COMPRESS_MAX_SENTS", "10")),
            # Skip per-chunk compression entirely — only budget-trim is used
            max_chars_per_chunk=LLM_CONTEXT_WINDOW_CHARS,
        )
    return _compressor

def get_eval_llm():
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance
    try:
        _llm_instance = get_insurance_llm(temperature=0)
        info = get_active_model_info()
        logger.info("[LLM] singleton created (backend=%s model=%s)", info.get("backend"), info.get("model"))
    except Exception as exc:
        logger.warning("[LLM] unavailable — falling back to regex-only: %s", exc)
        _llm_instance = None
    return _llm_instance


# ── Startup ────────────────────────────────────────────────────────────────────
from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(application):
    logger.info("[STARTUP] Pre-warming all singletons ...")
    try:
        vs  = get_vector_store()
        ss  = get_summary_store()
        kv  = get_kv_cache()
        cmp = get_compressor()
        llm = get_eval_llm()
        info = get_active_model_info()
        vs.warmup()
        logger.info(
            "[STARTUP] Ready — chunks=%d  summaries=%d  kv_entries=%d  llm=%s/%s",
            vs.count(), ss.count(), kv.stats()["live_entries"],
            info.get("backend"), info.get("model"),
        )
    except Exception as exc:
        logger.warning("[STARTUP] Pre-warm error (non-fatal): %s", exc)
    yield
    logger.info("[SHUTDOWN] Server process ending")


app = FastAPI(
    title="InsureHub RAG Evaluator",
    description="Evaluation dashboard API: metadata · raw chunks · RAGAS · timing",
    version="1.0.0",
    lifespan=_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────
class EvalRequest(BaseModel):
    query: str
    top_k: int = 8
    use_hybrid: bool = True
    use_reranker: bool = True
    generate_answer: bool = True
    run_ragas: bool = True

class ChunkInfo(BaseModel):
    chunk_index: int
    text: str
    char_count: int
    word_count: int
    similarity_score: float
    rerank_score: Optional[float]
    retrieval_method: str
    metadata: dict[str, Any]
    section: str

class RagasScores(BaseModel):
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    judge_model: str
    is_fallback: bool = False

class EvalResponse(BaseModel):
    query: str
    doc_metadata: dict[str, Any]
    chunk_metadata: list[dict[str, Any]]
    chunks: list[ChunkInfo]
    total_chunks_in_store: int
    ragas: Optional[RagasScores]
    ragas_per_chunk: list[dict[str, Any]]
    answer: Optional[str]
    sources: list[str]
    has_conflict: bool
    conflict_insurers: list[str]
    timing: dict[str, float]


# ══════════════════════════════════════════════════════════════════════════════
# YOUTUBE METADATA ENRICHMENT — LLM-generated topic / entity / chunk_type
# ══════════════════════════════════════════════════════════════════════════════

_YT_TOPICS = (
    "travel_insurance", "health_insurance", "life_insurance", "general_insurance",
    "motor_insurance", "home_insurance", "micro_insurance", "marine_insurance",
    "fire_insurance", "term_insurance", "ulip", "other",
)
_YT_CHUNK_TYPES = (
    "explanation", "definition", "example", "comparison",
    "faq", "news", "case_study", "general",
)


def _tag_youtube_doc_meta(full_text: str, url: str, video_title: str, llm: Any) -> dict:
    """
    Use LLM to generate document-level YouTube metadata:
      topic            — main insurance topic (e.g. travel_insurance)
      entity           — key company / country / product (e.g. qatar, hdfc)
      document_summary — 2–3 sentence summary of what the video covers

    Returns an empty dict if LLM is unavailable or fails.
    """
    if llm is None:
        return {}

    preview = full_text[:3000]
    prompt = (
        f"You are analyzing a YouTube video transcript about insurance.\n"
        f"Video URL: {url}\n"
        f"Video Title: {video_title}\n"
        f"Transcript (first 3000 chars):\n{preview}\n\n"
        f"Respond in strict JSON with exactly these keys:\n"
        f"  topic           — one of: {', '.join(_YT_TOPICS)}\n"
        f"  entity          — main company/country/product entity, or 'general'\n"
        f"  document_summary — 2-3 sentences summarising what the video covers\n\n"
        f"JSON only, no other text:"
    )
    try:
        raw = llm.invoke(prompt)
        raw = raw.content if hasattr(raw, "content") else str(raw)
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        data = json.loads(raw)
        return {
            "topic":            str(data.get("topic", "general")).strip().lower(),
            "entity":           str(data.get("entity", "general")).strip().lower(),
            "document_summary": str(data.get("document_summary", "")).strip(),
        }
    except Exception as exc:
        logger.warning("[TAG_YT_DOC] LLM doc-level tagging failed: %s", exc)
        return {}


def _tag_youtube_chunk_meta(doc, doc_meta: dict, llm: Any) -> None:
    """
    Add chunk_type, topic, entity to a single YouTube chunk in-place via LLM.
    Falls back to doc-level values when LLM is unavailable or fails.
    """
    fallback_topic  = doc_meta.get("topic",  "general")
    fallback_entity = doc_meta.get("entity", "general")

    if llm is None:
        doc.metadata.setdefault("chunk_type", "general")
        doc.metadata.setdefault("topic",      fallback_topic)
        doc.metadata.setdefault("entity",     fallback_entity)
        return

    text = doc.page_content[:500]
    prompt = (
        f"Classify this insurance video transcript chunk.\n\n"
        f"Chunk:\n{text}\n\n"
        f"Respond in strict JSON with exactly these keys:\n"
        f"  chunk_type — one of: {', '.join(_YT_CHUNK_TYPES)}\n"
        f"  topic      — one of: {', '.join(_YT_TOPICS)}\n"
        f"  entity     — main entity in this chunk (company/country/product) or 'general'\n\n"
        f"JSON only:"
    )
    try:
        raw = llm.invoke(prompt)
        raw = raw.content if hasattr(raw, "content") else str(raw)
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        data = json.loads(raw)
        doc.metadata["chunk_type"] = str(data.get("chunk_type", "general")).strip().lower()
        doc.metadata["topic"]      = str(data.get("topic",      fallback_topic)).strip().lower()
        doc.metadata["entity"]     = str(data.get("entity",     fallback_entity)).strip().lower()
    except Exception as exc:
        logger.debug("[TAG_YT_CHUNK] LLM chunk tagging failed: %s", exc)
        doc.metadata.setdefault("chunk_type", "general")
        doc.metadata.setdefault("topic",      fallback_topic)
        doc.metadata.setdefault("entity",     fallback_entity)


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION HELPER — fast regex first, parallel LLM only when needed
# ══════════════════════════════════════════════════════════════════════════════

def _classify_docs_metadata(docs: list, doc_type: str, doc_meta: dict, llm: Any) -> None:
    """
    Per-chunk classification: section (intent) + policy_type.

    FIX vs old version:
    - SectionChunker is now called with llm=None (zero LLM calls during chunking).
    - This function runs ONE pass after chunking.
    - For regular docs: force_llm=False — regex handles most chunks, LLM only
      for genuinely ambiguous ones. Typical 200-chunk doc → ~10-20 LLM calls.
    - For YouTube: force_llm=True but ALL calls run in a ThreadPoolExecutor
      so 200 YouTube chunks take max(t_single_call) instead of 200×t.
    - Chunk text is trimmed to 150 chars for classification — enough signal,
      much shorter prompt, ~3× faster per LLM call.
    """
    is_youtube = doc_type in ("youtube",) or any(
        "whisper" in str(d.metadata.get("source_type", "")).lower() or
        "youtube" in str(d.metadata.get("source_type", "")).lower()
        for d in docs
    )
    force_llm = is_youtube   # only force LLM for YouTube; regex handles docs

    # For regular docs, classify serially (regex fast-path skips most LLM calls)
    # For YouTube, classify in parallel (all chunks need LLM due to force_llm=True)
    if not is_youtube or llm is None:
        for doc in docs:
            _classify_one(doc, doc_type, doc_meta, llm, force_llm)
    else:
        # Parallel classification for YouTube chunks
        with ThreadPoolExecutor(max_workers=min(8, len(docs))) as pool:
            futures = {
                pool.submit(_classify_one, doc, doc_type, doc_meta, llm, True): doc
                for doc in docs
            }
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    logger.warning("[CLASSIFY] parallel chunk classification failed: %s", exc)


def _classify_one(doc, doc_type: str, doc_meta: dict, llm: Any, force_llm: bool) -> None:
    """Classify a single chunk in-place. Trims text to 150 chars for LLM speed."""
    # Use first 150 chars for classification — enough for section/policy signal
    text_for_classify = doc.page_content[:150]

    doc.metadata["section"] = classify_chunk_intent(
        text_for_classify,
        doc_type=doc_type,
        llm=llm,
        force_llm=force_llm,
    )

    chunk_policy = classify_chunk_policy_type(
        text_for_classify,
        llm=llm,
        force_llm=force_llm,
    )
    doc.metadata["policy_type"] = (
        chunk_policy if chunk_policy != "general"
        else doc_meta.get("policy_type", "general")
    )

    logger.debug(
        "[CLASSIFY] doc_type=%s force=%s | section=%s | policy_type=%s | text=%r",
        doc_type, force_llm,
        doc.metadata["section"], doc.metadata["policy_type"],
        doc.page_content[:60].replace("\n", " "),
    )


def _should_force_llm(doc_type: str, source_type: str = "") -> bool:
    source_type_lower = source_type.lower()
    return (
        doc_type in ("youtube", "general")
        or "whisper" in source_type_lower
        or "youtube" in source_type_lower
    )


# ══════════════════════════════════════════════════════════════════════════════
# RETRIEVAL THRESHOLDS  (module-level so both streaming and non-streaming share them)
# ══════════════════════════════════════════════════════════════════════════════
_MIN_RERANK_FOR_LLM = 0.08   # minimum rerank score to pass a chunk into LLM context
_LLM_GUARANTEED_TOP = 2      # always keep this many top chunks regardless of threshold

_QUERY_STOP_WORDS = {
    'what', 'is', 'are', 'the', 'a', 'an', 'in', 'of', 'for', 'how', 'does',
    'do', 'i', 'my', 'me', 'by', 'with', 'under', 'about', 'can', 'will',
    'which', 'when', 'where', 'to', 'and', 'or', 'at', 'this', 'that', 'it',
    'its', 'be', 'been', 'has', 'have', 'had', 'any', 'all', 'from', 'on',
    # domain-generic: present in virtually every insurance chunk, so not discriminating
    'insurance', 'insured', 'insurer', 'policy', 'policies', 'cover', 'coverage',
    'plan', 'claim', 'claims', 'benefits', 'benefit',
}

def _context_covers_query(query: str, docs: list) -> bool:
    """True if ≥1 chunk contains at least 1 discriminating keyword from the query.

    Filters both common English stop words and domain-generic insurance terms so that a
    chunk containing only 'insurance' doesn't falsely match a query about 'deductibles'.
    """
    import re
    query_terms = {w for w in re.findall(r'\b[a-z]{3,}\b', query.lower()) if w not in _QUERY_STOP_WORDS}
    if not query_terms:
        return True  # no discriminating terms — can't tell, assume covered
    for doc in docs:
        text = (doc.page_content if hasattr(doc, "page_content") else doc.get("text", "")).lower()
        chunk_terms = set(re.findall(r'\b[a-z]{3,}\b', text))
        if query_terms & chunk_terms:  # at least 1 specific term found
            return True
    return False

# ══════════════════════════════════════════════════════════════════════════════
# SOURCE INTENT + SHARED RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

_VIDEO_SIGNALS   = {"video", "youtube", "watch", "the video", "in the video",
                    "video says", "video mentioned", "talked about", "spoken",
                    "lecture", "clip", "webinar", "recording", "explained in video",
                    "as shown in", "mentioned in the video"}
_DOC_SIGNALS     = {"document", "pdf", "policy", "file", "article",
                    "handbook", "the document", "in the document", "policy document",
                    "as per the document", "according to the document"}
_COMBINE_SIGNALS = {"both", "together", "all sources", "across", "combine",
                    "and the video", "and the document", "as well as",
                    "from both", "video and document", "document and video",
                    "policy and video", "video and policy",
                    "including video", "and youtube", "youtube and",
                    "compare across", "from all", "summarize all",
                    "from videos and", "and from the"}

def _extract_cited_sources(answer: str, fallback_sources: list[str]) -> list[str]:
    """
    Parse citation tags from the LLM answer. Handles multiple formats the model may use:
      [Doc: filename, p.N]        — instructed format
      [Source: filename, Page: N] — Qwen's preferred format
      [Video: URL]                — video citations
    Falls back to fallback_sources if no citations found.
    """
    cited: list[str] = []
    seen: set[str] = set()

    # [Doc: filename, p.N] or [Doc: filename]
    for m in re.finditer(r'\[Doc:\s*([^\],]+?)(?:,\s*p\.?\s*[\w\-–]+)?\]', answer, re.IGNORECASE):
        src = m.group(1).strip()
        if src and src not in seen:
            seen.add(src); cited.append(src)

    # [Source: filename, Page: N] — Qwen model's native citation style
    for m in re.finditer(r'\[Source:\s*([^\],]+?)(?:,\s*Page:\s*[\w\-–]+)?\]', answer, re.IGNORECASE):
        src = m.group(1).strip()
        if src and src not in seen:
            seen.add(src); cited.append(src)

    # [Video: URL]
    for m in re.finditer(r'\[Video:\s*([^\]]+?)\]', answer, re.IGNORECASE):
        src = m.group(1).strip()
        if src and src not in seen:
            seen.add(src); cited.append(src)

    return cited if cited else fallback_sources


def _source_intent(query: str) -> str:
    q = query.lower()
    has_video   = any(sig in q for sig in _VIDEO_SIGNALS)
    has_doc     = any(sig in q for sig in _DOC_SIGNALS)
    has_combine = any(sig in q for sig in _COMBINE_SIGNALS)
    if has_combine or (has_video and has_doc):
        return "combined"
    if has_video:
        return "video"
    if has_doc:
        return "document"
    return "any"


_DEFINITIONAL_RE = re.compile(
    r'^(what\s+is|what\s+are|define|definition\s+of|explain|describe|meaning\s+of|tell\s+me\s+(about|what))\b',
    re.IGNORECASE,
)

def _is_definitional_query(query: str) -> bool:
    """True for 'What is X?', 'Define X', 'Explain X' style queries."""
    return bool(_DEFINITIONAL_RE.match(query.strip()))

def _strip_question_prefix(text: str) -> str:
    """Strip leading exam-question lines from chunk text.
    Handles OCR artifacts where multiple exam questions are packed into
    one long paragraph at the start of a chunk (continuation from prev page).
    Returns the trimmed text, or the original if no prefix was stripped.
    """
    lines = text.split('\n')
    start = 0
    for i, raw_line in enumerate(lines):
        l = raw_line.strip()
        if not l:
            continue
        words = l.split()
        # Long line with 3+ question marks = question parade in one paragraph
        if l.count('?') >= 3:
            start = i + 1
            continue
        # Normal short question line
        if 2 <= len(words) <= 35 and '?' in l:
            start = i + 1
            continue
        break  # first non-question line found
    if start == 0:
        return text
    trimmed = '\n'.join(lines[start:]).strip()
    return trimmed if trimmed else text


def _is_question_dump(text: str) -> bool:
    """True if a chunk is mostly exam/review questions rather than factual content.
    Cross-encoder rerankers falsely score these high when the query is also a question."""
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 8]
    if len(lines) < 2:
        return False

    def _is_q_line(l: str) -> bool:
        words = l.split()
        # Long line with 3+ question marks = question parade (OCR artifact)
        if l.count('?') >= 3:
            return True
        if len(words) > 35 or len(words) < 2:
            return False
        if '?' in l:
            return True
        return False

    # Opening-block check: if the first 5 lines are mostly questions, the chunk
    # is a question dump even if content follows (full-ratio would be diluted).
    head = lines[:5]
    if len(head) >= 2 and sum(1 for l in head if _is_q_line(l)) / len(head) >= 0.6:
        return True

    # Full-chunk ratio fallback
    q_count = sum(1 for l in lines if _is_q_line(l))
    return (q_count / len(lines)) >= 0.45

_OUTLINE_STRUCTURAL_HEADERS = re.compile(
    r'\b(LEARNING\s+OBJECTIVES?|LESSON\s+OUTLINE|TABLE\s+OF\s+CONTENTS|'
    r'LEARNING\s+OUTCOMES?|UNIT\s+OVERVIEW|CHAPTER\s+OVERVIEW|'
    r'COURSE\s+OUTLINE|CHAPTER\s+OUTLINE|UNIT\s+OUTLINE|'
    r'TOPICS?\s+COVERED|SCOPE\s+AND\s+SEQUENCE)\b',
    re.IGNORECASE,
)

_OBJECTIVE_LINE_RE = re.compile(
    r'^\s*\d+[\.\)]\s+'
    r'(Meaning\s+of|Understanding\s+of?|Understand\s|'
    r'Definition\s+of|Importance\s+of|Role\s+of|'
    r'Concept\s+of|Differentiat|Distinguish|'
    r'How\s+(to\s+)?|What\s+(is|are)\s|Explain\s|'
    r'Identify\s|Describe\s|Study\s+of)',
    re.IGNORECASE,
)


def _is_lesson_outline_chunk(text: str) -> bool:
    """True if a chunk is a lesson/chapter overview: structural headers
    (LEARNING OBJECTIVES, LESSON OUTLINE) plus numbered objective lines.
    These score high on keyword matching but contain no actual definitions.
    General: works across any textbook PDF regardless of subject."""
    if not _OUTLINE_STRUCTURAL_HEADERS.search(text):
        return False
    lines = text.split('\n')
    obj_lines = sum(1 for l in lines if _OBJECTIVE_LINE_RE.match(l))
    return obj_lines >= 2


def _is_youtube_doc(meta: dict) -> bool:
    return (
        meta.get("doc_type") == "youtube"
        or "youtube" in str(meta.get("source_type", "")).lower()
        or "whisper" in str(meta.get("source_type", "")).lower()
        or "youtube.com" in str(meta.get("source", "")).lower()
        or "youtu.be" in str(meta.get("source", "")).lower()
    )

def _retrieve(req: "EvalRequest") -> dict:
    vs = get_vector_store()
    ss = get_summary_store()
    t0 = time.perf_counter()

    # Clean query: strip ?, !, ;, : so BM25 tokens match document tokens exactly.
    # The original req.query is kept for LLM prompts, cache keys, and classification.
    search_query = _clean_query(req.query)
    intent_topic = _extract_query_intent(req.query)

    intent = _source_intent(req.query)
    logger.info(
        "[RETRIEVE] intent=%s  original=%r  search=%r  topic=%r",
        intent, req.query[:60], search_query[:60], intent_topic[:40],
    )

    # ── Stage 1: semantic summary search ──────────────────────────────────────
    # Embed the query → compare with summary embeddings → identify the most
    # relevant source documents before touching the chunk index.
    # Summaries are short (one entry per source), so this is very fast even
    # for large knowledge bases.
    yt_sources_in_store: list[str] = []
    yt_exists                      = False
    summary_context                = ""
    source_filter_for_chunks: Optional[dict] = None

    if ss.count() > 0:
        # Cheap in-memory metadata scan — just to know which sources are YouTube.
        all_summaries       = ss.list_all()
        yt_sources_in_store = [
            s["metadata"].get("source", "")
            for s in all_summaries
            if _is_youtube_doc(s["metadata"])
        ]
        yt_exists = bool(yt_sources_in_store)

        # Semantic stage 1: top-7 summaries ranked by embedding similarity.
        # Fixed small k (not ss.count()) so we only process the most relevant
        # summaries, not every one in the store.
        stage1_k    = min(ss.count(), 7)
        stage1_docs = ss.search(search_query, top_k=stage1_k)

        yt_sum_docs  = [d for d in stage1_docs if _is_youtube_doc(d.metadata)]
        pdf_sum_docs = [d for d in stage1_docs if not _is_youtube_doc(d.metadata)]

        if intent == "video":
            selected = yt_sum_docs[:3] + pdf_sum_docs[:1]
        elif intent == "combined":
            selected = pdf_sum_docs[:2] + yt_sum_docs[:2]
        elif intent == "document":
            selected = pdf_sum_docs[:5]
        else:  # "any"
            selected = pdf_sum_docs[:3] + (yt_sum_docs[:2] if yt_exists else [])

        if selected:
            parts = []
            for doc in selected:
                src     = doc.metadata.get("source", "?")
                dtype   = doc.metadata.get("doc_type", "document")
                insurer = doc.metadata.get("insurer", "UNKNOWN")
                parts.append(f"[Summary | {dtype} | {src} | {insurer}]\n{doc.page_content}")
            summary_context = "\n\n".join(parts)

            # Unified source filter for ALL intents: chunk search is scoped to the
            # top semantically-matched sources from stage 1.
            # Unsummarized sources are always included so newly-added docs are
            # never missed even before their summaries are indexed.
            top_src_names  = [d.metadata.get("source", "") for d in selected if d.metadata.get("source")]
            all_chunk_srcs = set(vs.list_sources())
            unsummarized   = list(all_chunk_srcs - set(ss.list_sources()))
            allowed        = list(dict.fromkeys(top_src_names + unsummarized))
            if allowed and set(allowed) < all_chunk_srcs:
                source_filter_for_chunks = {"source": {"$in": allowed}}

    query_meta = classify_query(req.query, llm=None)
    routed_sources = _route_to_documents(req.query, vs.list_sources())
    if routed_sources:
        logger.info("[RETRIEVE] keyword route → %s", routed_sources)

    policy_filter = build_metadata_filter(
        query_meta,
        routed_sources,
        policy_confidence_threshold=0.75,
    )

    # Priority: explicit keyword route > semantic source filter > policy filter.
    # Keyword route is the most specific signal (user named a document/policy),
    # so it overrides the semantic source filter from stage 1.
    if routed_sources and intent not in ("combined", "any"):
        filter_meta: Optional[dict] = RAGPipeline._source_filter(routed_sources)
    elif source_filter_for_chunks:
        filter_meta = source_filter_for_chunks
    elif intent in ("combined", "any"):
        filter_meta = None
    else:
        filter_meta = policy_filter

    t_ret = time.perf_counter()
    raw_docs = vs.search(
        query=search_query,
        top_k=max(req.top_k * 5, 30),  # wider pool so reranker has more candidates
        filter_metadata=filter_meta,
        use_hybrid=req.use_hybrid,
        use_reranker=False,
    )
    if not raw_docs and filter_meta is not None:
        logger.warning("[RETRIEVE] filtered search returned 0 — retrying unfiltered")
        raw_docs = vs.search(
            query=search_query,
            top_k=req.top_k * 2,
            filter_metadata=None,
            use_hybrid=req.use_hybrid,
            use_reranker=False,
        )

    # Stage-1 source guarantee: the main ANN search is capped at top-30 across
    # all chunks. When the store has hundreds of PDF chunks, a small YouTube (or any
    # low-frequency) source can fall below that ceiling even if it IS relevant.
    # For every source the Stage-1 summary search deemed relevant, run a tiny
    # per-source search so it always has at least one representative in the pool.
    # This applies equally to PDFs and video transcripts — no special-case logic.
    if selected:
        in_pool_srcs = {d.metadata.get("source", "") for d in raw_docs}
        for summary_doc in selected:
            src = summary_doc.metadata.get("source", "")
            if src and src not in in_pool_srcs:
                boost = vs.search(
                    query=search_query,
                    top_k=2,
                    filter_metadata={"source": {"$eq": src}},
                    use_hybrid=req.use_hybrid,
                    use_reranker=False,
                )
                if boost:
                    existing_keys = {d.page_content[:80] for d in raw_docs}
                    raw_docs = raw_docs + [d for d in boost if d.page_content[:80] not in existing_keys]
                    in_pool_srcs.add(src)
                    logger.info("[RETRIEVE] stage-1 boost: added %d chunk(s) from %r", len(boost), src)

    # Definitional query boost: for "What is X?" / "Define X" queries, pull in
    # chunks from the definitions section that the main search may have ranked low
    # because BM25 treats them the same as any chunk containing "insurance".
    if _is_definitional_query(req.query):
        def_extra = vs.search(
            query=search_query,
            top_k=min(req.top_k, 5),
            filter_metadata={"section": {"$eq": "definitions"}},
            use_hybrid=req.use_hybrid,
            use_reranker=False,
        )
        if def_extra:
            existing_keys = {d.page_content[:80] for d in raw_docs}
            raw_docs = raw_docs + [d for d in def_extra if d.page_content[:80] not in existing_keys]

    retrieval_ms = round((time.perf_counter() - t_ret) * 1000, 1)

    def _score(doc) -> float:
        return doc.metadata.get("rerank_score", doc.metadata.get("similarity", 0.0))
    def _fingerprint(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())[:200]

    exact: dict = {}
    for doc in raw_docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page"), doc.page_content)
        if key not in exact or _score(doc) > _score(exact[key]):
            exact[key] = doc
    fp_seen: dict = {}
    for doc in exact.values():
        fp = _fingerprint(doc.page_content)
        if fp not in fp_seen or _score(doc) > _score(fp_seen[fp]):
            fp_seen[fp] = doc
    deduped = list(fp_seen.values())

    # Strip leading question-paragraph prefixes then filter pure question dumps.
    # OCR artifacts pack multiple exam questions into one long line at the start
    # of a chunk (continuation from the previous page). We strip that prefix so
    # the rest of the chunk (useful content) survives. Pure question-dump chunks
    # (no content left after stripping) are dropped entirely.
    from langchain_core.documents import Document as _Doc
    stripped = []
    for d in deduped:
        clean = _strip_question_prefix(d.page_content)
        stripped.append(_Doc(page_content=clean, metadata=d.metadata) if clean != d.page_content else d)
    deduped = stripped

    before_filter = len(deduped)
    deduped = [d for d in deduped if not _is_question_dump(d.page_content)]
    if len(deduped) < before_filter:
        logger.info("[RETRIEVE] dropped %d question-dump chunks", before_filter - len(deduped))

    before_filter = len(deduped)
    deduped = [d for d in deduped if not _is_lesson_outline_chunk(d.page_content)]
    if len(deduped) < before_filter:
        logger.info("[RETRIEVE] dropped %d lesson-outline chunks", before_filter - len(deduped))

    if req.use_reranker and len(deduped) > 1:
        deduped = vs.rerank_documents(req.query, deduped, top_k=len(deduped))
        score_key = lambda d: d.metadata.get("rerank_score", 0)
    else:
        score_key = lambda d: d.metadata.get("rerank_score", d.metadata.get("similarity", 0))

    # All documents (PDF and video transcripts) ranked by the same score.
    # Source diversity: ensure at least one chunk per source in top-K so no
    # single document monopolises all context slots.
    all_ranked = sorted(deduped, key=score_key, reverse=True)
    best_per_src: dict = {}
    for d in all_ranked:
        src = d.metadata.get("source", "")
        if src and src not in best_per_src:
            best_per_src[src] = d
    top_srcs = sorted(best_per_src, key=lambda s: score_key(best_per_src[s]), reverse=True)
    max_diversity = min(len(top_srcs), max(1, req.top_k // 2))
    diversity_chunks = [best_per_src[s] for s in top_srcs[:max_diversity]]
    diversity_ids = {id(d) for d in diversity_chunks}
    filler = [d for d in all_ranked if id(d) not in diversity_ids]
    raw_docs = sorted(
        diversity_chunks + filler[: max(0, req.top_k - len(diversity_chunks))],
        key=score_key, reverse=True,
    )

    # FIX: only compress when total context exceeds LLM window — not always
    total_chars = sum(len(d.page_content) for d in raw_docs)
    if total_chars > LLM_CONTEXT_WINDOW_CHARS:
        logger.info("[RETRIEVE] compressing context (%d > %d chars)", total_chars, LLM_CONTEXT_WINDOW_CHARS)
        raw_docs = get_compressor().compress_to_budget(req.query, list(raw_docs), LLM_CONTEXT_WINDOW_CHARS)

    # Uniform relevance gate — same threshold for every chunk regardless of source type.
    raw_docs_for_llm = [d for d in raw_docs if d.metadata.get("rerank_score", 1.0) >= _MIN_RERANK_FOR_LLM]

    # Always keep the top-N chunks so the LLM always has context even on weak queries.
    by_rerank = sorted(raw_docs, key=lambda d: d.metadata.get("rerank_score", 0), reverse=True)
    guaranteed_keys = {id(d) for d in raw_docs_for_llm}
    for d in by_rerank[:min(_LLM_GUARANTEED_TOP, len(by_rerank))]:
        if id(d) not in guaranteed_keys:
            raw_docs_for_llm.append(d)
            guaranteed_keys.add(id(d))

    # Re-sort by rerank so highest-confidence chunks appear first in context
    raw_docs_for_llm.sort(key=lambda d: d.metadata.get("rerank_score", 0), reverse=True)
    # Attach flag for UI display
    llm_ids = {id(d) for d in raw_docs_for_llm}
    for d in raw_docs:
        d.metadata["_include_in_context"] = id(d) in llm_ids

    return {
        "raw_docs":         raw_docs,          # all retrieved chunks (for UI display)
        "raw_docs_for_llm": raw_docs_for_llm,  # rerank-filtered subset (for LLM context)
        "summary_context":  summary_context,
        "filter_meta":      filter_meta,
        "query_meta":       query_meta,
        "retrieval_ms":     retrieval_ms,
        "intent":           intent,
    }


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/eval/upload", summary="Upload & ingest a document into the vector store")
async def eval_upload(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "doc.pdf")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=_SAFE_TMP_ROOT) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    t0 = time.perf_counter()
    try:
        docs = load_document(tmp_path, file.filename or "document")
    except Exception as exc:
        traceback.print_exc()
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail=f"Document loading failed: {exc}")

    if not docs:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail="No extractable content found in document.")

    llm    = get_eval_llm()
    fname  = file.filename or "document"
    preview    = " ".join(d.page_content for d in docs[:1])[:5000]
    extra_text = " ".join(d.page_content for d in docs[1:4])[:5000]
    doc_type   = classify_document_type(fname, preview, extra_text)
    doc_meta   = tag_document(fname, preview, extra_text=extra_text, doc_type=doc_type, llm=llm)
    doc_meta["doc_type"] = doc_type
    logger.info("[UPLOAD] '%s' → doc_type=%s, policy_type=%s", fname, doc_type, doc_meta.get("policy_type"))

    is_youtube_doc = doc_type == "youtube" or any(
        d.metadata.get("source_type", "") in ("youtube_transcript", "whisper") for d in docs
    )

    for d in docs:
        d.metadata["doc_type"] = doc_type
        if is_youtube_doc:
            d.metadata.setdefault("source_type", "youtube_transcript")

    # FIX: call SectionChunker with llm=None — zero LLM calls during chunking.
    # Classification happens in _classify_docs_metadata() below with
    # parallelism for YouTube and regex fast-path for regular docs.
    try:
        pipeline = get_rag_pipeline()
        chunks = pipeline.chunker.split_documents(docs, doc_type=doc_type, llm=None)
        logger.info("[UPLOAD] SectionChunker → %d chunks (llm=None, fast)", len(chunks))
    except Exception as exc:
        traceback.print_exc()
        logger.warning("[UPLOAD] SectionChunker failed (%s), using raw pages", exc)
        chunks = docs

    for chunk in chunks:
        chunk.metadata.setdefault("filename",    fname)
        chunk.metadata.setdefault("source",      fname)
        chunk.metadata.setdefault("source_type", "document")
        chunk.metadata["doc_type"] = doc_type
        chunk.metadata["insurer"]  = chunk.metadata.get("insurer") or doc_meta.get("insurer", "UNKNOWN")

    # ONE classification pass — regex fast-path for docs, parallel LLM for YouTube
    t_classify = time.perf_counter()
    _classify_docs_metadata(chunks, doc_type, doc_meta, llm)
    classify_ms = round((time.perf_counter() - t_classify) * 1000)
    logger.info("[UPLOAD] classification done in %dms for %d chunks", classify_ms, len(chunks))

    vs  = get_vector_store()
    ids = vs.add_documents(chunks)
    elapsed = (time.perf_counter() - t0) * 1000
    os.unlink(tmp_path)

    assigned_policy_types = list({c.metadata.get("policy_type", "general") for c in chunks})
    assigned_sections     = list({c.metadata.get("section", "general") for c in chunks})
    general_count = sum(1 for c in chunks if c.metadata.get("policy_type", "general") == "general")

    logger.info(
        "[UPLOAD] done — %d chunks | policy_types=%s | %d/%d still 'general' | total=%.0fms (classify=%dms)",
        len(chunks), assigned_policy_types, general_count, len(chunks), elapsed, classify_ms,
    )

    try:
        summary_text = generate_summary(docs, source=fname, doc_meta=doc_meta, llm=llm)
        get_summary_store().upsert(
            source=fname,
            summary_text=summary_text,
            metadata={
                "source_type":  docs[0].metadata.get("source_type", "document") if docs else "document",
                "doc_type":     doc_type,
                "insurer":      doc_meta.get("insurer", "UNKNOWN"),
                "policy_type":  doc_meta.get("policy_type", "general"),
                "chunk_count":  len(ids),
                "title":        fname,
                "summary_type": "document",
            },
        )
    except Exception as exc:
        logger.warning("[UPLOAD] summary generation failed: %s", exc)

    return {
        "status": "ok",
        "filename": fname,
        "doc_type": doc_type,
        "chunks_added": len(ids),
        "total_in_store": vs.count(),
        "doc_metadata": doc_meta,
        "assigned_policy_types": assigned_policy_types,
        "assigned_sections": assigned_sections,
        "chunks_still_general_policy_type": general_count,
        "ingest_ms": round(elapsed, 1),
        "classify_ms": classify_ms,
    }


@app.get("/eval/documents", summary="List all ingested documents")
def eval_documents():
    vs = get_vector_store()
    return {"sources": vs.list_sources(), "filenames": vs.list_filenames(), "total_chunks": vs.count()}


class DeleteRequest(BaseModel):
    source: str

@app.post("/eval/delete", summary="Remove a document from the store by source key")
def eval_delete(req: DeleteRequest):
    vs = get_vector_store()
    vs.delete_by_source(req.source)
    get_summary_store().delete(req.source)
    return {"status": "deleted", "source": req.source, "remaining_chunks": vs.count()}

@app.delete("/eval/documents/{source}", summary="Remove a document by filename")
def eval_delete_path(source: str):
    vs = get_vector_store()
    vs.delete_by_source(source)
    get_summary_store().delete(source)
    return {"status": "deleted", "source": source, "remaining_chunks": vs.count()}


# ══════════════════════════════════════════════════════════════════════════════
# INGEST YOUTUBE
# ══════════════════════════════════════════════════════════════════════════════

class VideoIngestRequest(BaseModel):
    url: str

@app.post("/eval/ingest-video", summary="Ingest a YouTube video transcript")
def eval_ingest_video(req: VideoIngestRequest):
    url = req.url.strip()
    if not is_youtube_url(url):
        raise HTTPException(status_code=422, detail="URL does not appear to be a YouTube link.")
    t0 = time.perf_counter()
    try:
        docs = _load_youtube(url)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Failed to load YouTube transcript: {exc}")
    if not docs:
        raise HTTPException(status_code=422, detail="No transcript content could be extracted.")

    # ── Extract transcript metadata from loader ────────────────────────────────
    llm           = get_eval_llm()
    video_id      = docs[0].metadata.get("video_id") or "unknown"
    video_title   = docs[0].metadata.get("video_title") or f"YouTube video ({video_id})"
    detected_lang = docs[0].metadata.get("language", "unknown")
    source_type   = docs[0].metadata.get("source_type", "youtube_transcript")
    doc_type      = "youtube"
    # Use video title as synthetic filename for readability in UI
    safe_title    = re.sub(r"[^\w\s-]", "", video_title)[:60].strip().replace(" ", "_")
    syn_name      = f"youtube_{video_id}_{safe_title}.txt" if safe_title else f"youtube_{video_id}.txt"

    # ── Warn if the transcript is not English ─────────────────────────────────
    lang_warning = ""
    if detected_lang not in ("en", "unknown"):
        lang_warning = (
            f"WARNING: The transcript appears to be in '{detected_lang}' (non-English). "
            f"English queries may not retrieve content from this video. "
            f"Consider re-ingesting with an English-language video."
        )
        logger.warning("[INGEST-VIDEO] %s", lang_warning)

    # ── Document-level tagging (same pipeline as PDFs) ───────────────────────
    preview    = docs[0].page_content[:5000]
    extra_text = docs[0].page_content[5000:10000]
    doc_meta   = tag_document(syn_name, preview, extra_text=extra_text, doc_type=doc_type, llm=llm)
    doc_meta["doc_type"] = doc_type

    logger.info(
        "[INGEST-VIDEO] doc-level tags — title=%r lang=%s policy_type=%s insurer=%s",
        video_title, detected_lang,
        doc_meta.get("policy_type", "?"), doc_meta.get("insurer", "?"),
    )

    # ── Propagate doc-level fields to raw docs ────────────────────────────────
    for doc in docs:
        doc.metadata.setdefault("filename", syn_name)
        doc.metadata["source"]       = url
        doc.metadata["video_title"]  = video_title
        doc.metadata["language"]     = detected_lang
        doc.metadata["insurer"]      = doc_meta.get("insurer", "UNKNOWN")
        doc.metadata["doc_type"]     = doc_type  # kept for UI display only

    # ── Semantic chunking — same path as every other document ─────────────────
    try:
        pipeline = get_rag_pipeline()
        chunks = pipeline.chunker.split_documents(docs, doc_type="policy_document", llm=None)
        logger.info("[INGEST-VIDEO] SemanticChunker → %d chunks", len(chunks))
    except Exception as exc:
        traceback.print_exc()
        logger.warning("[INGEST-VIDEO] SemanticChunker failed (%s), using raw docs", exc)
        chunks = docs

    # ── Propagate doc-level metadata to every chunk ──────────────────────────
    for chunk in chunks:
        chunk.metadata.setdefault("filename", syn_name)
        chunk.metadata["source"]      = url
        chunk.metadata["video_title"] = video_title
        chunk.metadata["language"]    = detected_lang
        chunk.metadata["insurer"]     = doc_meta.get("insurer", "UNKNOWN")
        chunk.metadata["doc_type"]    = doc_type  # kept for UI display only
        chunk.metadata.setdefault("source_type", source_type)

    # ── Classification — identical to PDF path ────────────────────────────────
    t_classify = time.perf_counter()
    _classify_docs_metadata(chunks, "policy_document", doc_meta, llm)
    classify_ms = round((time.perf_counter() - t_classify) * 1000)
    logger.info("[INGEST-VIDEO] classification done in %dms for %d chunks", classify_ms, len(chunks))

    # ── Store in vector store ─────────────────────────────────────────────────
    vs  = get_vector_store()
    ids = vs.add_documents(chunks)
    elapsed = (time.perf_counter() - t0) * 1000

    assigned_policy_types = list({c.metadata.get("policy_type", "general") for c in chunks})
    assigned_sections     = list({c.metadata.get("section", "general") for c in chunks})
    assigned_chunk_types  = list({c.metadata.get("chunk_type", "general") for c in chunks})
    general_count = sum(1 for c in chunks if c.metadata.get("policy_type", "general") == "general")

    # ── Summary store ─────────────────────────────────────────────────────────
    try:
        summary_text = generate_summary(docs, source=url, doc_meta=doc_meta, llm=llm)
        get_summary_store().upsert(
            source=url,
            summary_text=summary_text,
            metadata={
                "source_type":      source_type,
                "doc_type":         doc_type,
                "insurer":          doc_meta.get("insurer",  "UNKNOWN"),
                "policy_type":      doc_meta.get("policy_type", "general"),
                "topic":            doc_meta.get("topic",    "general"),
                "entity":           doc_meta.get("entity",   "general"),
                "video_title":      video_title,
                "language":         detected_lang,
                "chunk_count":      len(ids),
                "title":            syn_name,
                "video_id":         video_id,
                "summary_type":     "video",
            },
        )
    except Exception as exc:
        logger.warning("[INGEST-VIDEO] summary failed: %s", exc)

    response = {
        "status":         "ok",
        "filename":       syn_name,
        "url":            url,
        "video_id":       video_id,
        "video_title":    video_title,
        "language":       detected_lang,
        "source_type":    source_type,
        "chunks_added":   len(ids),
        "total_in_store": vs.count(),
        "doc_metadata":   doc_meta,
        "assigned_policy_types":  assigned_policy_types,
        "assigned_sections":      assigned_sections,
        "assigned_chunk_types":   assigned_chunk_types,
        "chunks_still_general_policy_type": general_count,
        "ingest_ms":   round(elapsed, 1),
        "classify_ms": classify_ms,
    }
    if lang_warning:
        response["language_warning"] = lang_warning
    return response


# ══════════════════════════════════════════════════════════════════════════════
# INGEST WEBPAGE
# ══════════════════════════════════════════════════════════════════════════════

class WebpageIngestRequest(BaseModel):
    url: str

@app.post("/eval/ingest-webpage", summary="Scrape a web page and ingest it")
def eval_ingest_webpage(req: WebpageIngestRequest):
    url = req.url.strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="Please provide a valid http(s) URL.")
    if is_youtube_url(url):
        raise HTTPException(status_code=422, detail="Use /eval/ingest-video for YouTube links.")
    t0 = time.perf_counter()
    try:
        docs = load_url_advanced(url)
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=422, detail=f"Failed to fetch web page: {exc}")
    if not docs:
        raise HTTPException(status_code=422, detail="No extractable text content found on this page.")

    llm         = get_eval_llm()
    title       = docs[0].metadata.get("title") or url
    source_type = docs[0].metadata.get("source_type", "web")
    slug        = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())[:60].strip("_") or "page"
    syn_name    = f"webpage_{slug}.txt"
    preview     = docs[0].page_content[:5000]
    extra_text  = docs[0].page_content[5000:10000]
    doc_type    = classify_document_type(syn_name, preview, extra_text)
    doc_meta    = tag_document(syn_name, preview, extra_text=extra_text, doc_type=doc_type, llm=llm)
    doc_meta["doc_type"] = doc_type

    for doc in docs:
        doc.metadata.setdefault("filename", syn_name)
        doc.metadata["source"]      = url
        doc.metadata["source_url"]  = url
        doc.metadata["insurer"]     = doc_meta.get("insurer", "UNKNOWN")
        doc.metadata["doc_type"]    = doc_type
        doc.metadata.setdefault("source_type", source_type)

    try:
        pipeline = get_rag_pipeline()
        chunks = pipeline.chunker.split_documents(docs, doc_type=doc_type, llm=None)
        logger.info("[INGEST-WEBPAGE] SectionChunker → %d chunks (llm=None, fast)", len(chunks))
    except Exception as exc:
        traceback.print_exc()
        logger.warning("[INGEST-WEBPAGE] SectionChunker failed (%s), using raw docs", exc)
        chunks = docs

    for chunk in chunks:
        chunk.metadata.setdefault("filename", syn_name)
        chunk.metadata["source"]     = url
        chunk.metadata["source_url"] = url
        chunk.metadata["insurer"]    = doc_meta.get("insurer", "UNKNOWN")
        chunk.metadata["doc_type"]   = doc_type
        chunk.metadata.setdefault("source_type", source_type)

    t_classify = time.perf_counter()
    _classify_docs_metadata(chunks, doc_type, doc_meta, llm)
    classify_ms = round((time.perf_counter() - t_classify) * 1000)

    vs  = get_vector_store()
    ids = vs.add_documents(chunks)
    elapsed = (time.perf_counter() - t0) * 1000

    assigned_policy_types = list({c.metadata.get("policy_type", "general") for c in chunks})
    assigned_sections     = list({c.metadata.get("section", "general") for c in chunks})
    general_count = sum(1 for c in chunks if c.metadata.get("policy_type", "general") == "general")

    try:
        summary_text = generate_summary(docs, source=url, doc_meta=doc_meta, llm=llm)
        get_summary_store().upsert(
            source=url,
            summary_text=summary_text,
            metadata={
                "source_type": source_type, "doc_type": doc_type,
                "insurer": doc_meta.get("insurer", "UNKNOWN"),
                "policy_type": doc_meta.get("policy_type", "general"),
                "chunk_count": len(ids), "title": title, "summary_type": "webpage",
            },
        )
    except Exception as exc:
        logger.warning("[INGEST-WEBPAGE] summary failed: %s", exc)

    return {
        "status": "ok", "filename": syn_name, "url": url, "title": title,
        "source_type": source_type, "doc_type": doc_type,
        "chunks_added": len(ids), "total_in_store": vs.count(), "doc_metadata": doc_meta,
        "assigned_policy_types": assigned_policy_types, "assigned_sections": assigned_sections,
        "chunks_still_general_policy_type": general_count,
        "ingest_ms": round(elapsed, 1), "classify_ms": classify_ms,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/eval/query", response_model=EvalResponse, summary="Run full RAG evaluation")
def eval_query(req: EvalRequest):
    vs = get_vector_store()
    if vs.count() == 0:
        raise HTTPException(status_code=400, detail="Vector store is empty. Upload at least one document first.")

    kv = get_kv_cache()
    current_sources = vs.list_sources()
    cache_key = kv.make_key(
        req.query, req.top_k, req.use_hybrid, req.use_reranker,
        req.generate_answer, req.run_ragas, current_sources,
    )
    cached = kv.get(cache_key)
    _query_emb = None
    if cached is None:
        try:
            _query_emb = get_compressor()._model.encode(
                [req.query], normalize_embeddings=True, show_progress_bar=False
            )[0]
            sem_hit = kv.semantic_get(_query_emb)
            # Reject semantic hits where the cached entry has no answer but we need one —
            # a generate_answer=false result would otherwise poison generate_answer=true calls.
            if sem_hit is not None and req.generate_answer and not sem_hit.get("answer"):
                sem_hit = None
            cached = sem_hit
        except Exception as _sem_exc:
            logger.debug("[EVAL] semantic cache lookup failed: %s", _sem_exc)
    if cached is not None:
        cached["doc_metadata"]["from_cache"] = True
        return EvalResponse(**cached)

    t_start = time.perf_counter()
    llm = get_eval_llm()

    ret             = _retrieve(req)
    raw_docs        = ret["raw_docs"]
    raw_docs_for_llm = ret["raw_docs_for_llm"]
    summary_context = ret["summary_context"]
    filter_meta     = ret["filter_meta"]
    query_meta      = ret["query_meta"]
    retrieval_ms    = ret["retrieval_ms"]

    # Only show the chunks that were actually passed to the LLM.
    chunks: list[ChunkInfo] = []
    for idx, doc in enumerate(raw_docs_for_llm):
        meta = dict(doc.metadata)
        section = meta.get("section") or classify_chunk_intent(
            doc.page_content[:150], doc_type=meta.get("doc_type", "policy_document"), llm=None
        )
        sim    = float(meta.get("similarity", 0.0))
        rerank = float(meta["rerank_score"]) if "rerank_score" in meta else None
        chunks.append(ChunkInfo(
            chunk_index=idx, text=doc.page_content,
            char_count=len(doc.page_content), word_count=len(doc.page_content.split()),
            similarity_score=round(sim, 4),
            rerank_score=round(rerank, 4) if rerank is not None else None,
            retrieval_method=meta.get("retrieval_method", "dense"),
            metadata=meta, section=section,
        ))

    seen_sources: set[str] = set()
    doc_metadata: dict[str, Any] = {
        "query_insurer_hint":      query_meta.get("insurer"),
        "query_policy_type_hint":  query_meta.get("policy_type"),
        "metadata_filter_applied": filter_meta,
        "sources_retrieved":       [],
        "sections_hit":            [],
        "policy_types_hit":        [],
        "retrieval_method":        raw_docs[0].metadata.get("retrieval_method", "dense") if raw_docs else "n/a",
        "total_chunks_in_store":   vs.count(),
        "chunks_retrieved":        len(chunks),
    }
    sections_seen: set[str] = set()
    policy_types_seen: set[str] = set()
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        if src not in seen_sources:
            seen_sources.add(src)
            doc_metadata["sources_retrieved"].append(src)
        if c.section not in sections_seen:
            sections_seen.add(c.section); doc_metadata["sections_hit"].append(c.section)
        pt = c.metadata.get("policy_type", "general")
        if pt not in policy_types_seen:
            policy_types_seen.add(pt); doc_metadata["policy_types_hit"].append(pt)

    chunk_metadata = [
        {
            "chunk_index": c.chunk_index, "source": c.metadata.get("source", "unknown"),
            "filename": c.metadata.get("filename", "unknown"), "page": c.metadata.get("page"),
            "section": c.section, "insurer": c.metadata.get("insurer", "UNKNOWN"),
            "policy_type": c.metadata.get("policy_type", "general"),
            "doc_type": c.metadata.get("doc_type", "general"),
            "source_type": c.metadata.get("source_type", ""),
            "similarity_score": c.similarity_score, "rerank_score": c.rerank_score,
            "retrieval_method": c.retrieval_method,
            "word_count": c.word_count, "char_count": c.char_count,
        }
        for c in chunks
    ]

    has_conflict, conflict_insurers = detect_conflict(raw_docs)
    answer: Optional[str] = None
    sources: list[str] = []
    llm_ms = 0.0

    if req.generate_answer:
        t_llm_start = time.perf_counter()
        try:
            sources   = _sources_from_chunks(raw_docs_for_llm)
            context   = _build_structured_context(raw_docs_for_llm, max_chars=MAX_CONTEXT_CHARS)
            cond_hint = _extract_condition_hint(raw_docs_for_llm)
            has_conflict_local, conflict_ins = detect_conflict(raw_docs_for_llm)
            conflict_hint = (
                f"The context contains multiple insurers ({', '.join(sorted(conflict_ins))}). Keep facts separate."
                if has_conflict_local else ""
            )
            summary_section = (
                f"DOCUMENT SUMMARIES:\n{summary_context}\n\n" if summary_context else ""
            )
            _conflict_note = f"\nNOTE: {conflict_hint}" if conflict_hint else ""
            _cond_note     = f"\nCONDITIONAL CLAUSES: {cond_hint}" if cond_hint else ""
            unique_srcs = list(dict.fromkeys(
                d.metadata.get("source", "") for d in raw_docs_for_llm if d.metadata.get("source")
            ))
            src_list = ", ".join(f'"{s}"' for s in unique_srcs)
            ctx_covered = _context_covers_query(req.query, raw_docs_for_llm)
            _fallback_note = (
                "\nIMPORTANT: The retrieved document chunks do NOT contain explicit information "
                "about this specific topic. Provide a helpful, accurate general insurance "
                "explanation from your training knowledge and begin your answer with: "
                "'General knowledge (not from uploaded documents): '"
                if not ctx_covered else ""
            )
            citation_prompt = (
                f"You are an insurance document assistant. "
                f"The CONTEXT below contains chunks from these sources: {src_list}.\n"
                f"RULES:\n"
                f"1. If the CONTEXT contains relevant information: answer using ONLY the CONTEXT "
                f"and cite every fact as [Doc: filename, p.N].\n"
                f"2. If the CONTEXT does NOT explicitly cover the question: provide a clear, "
                f"accurate general insurance explanation from your training knowledge and label "
                f"it 'General knowledge (not from uploaded documents): '.\n"
                f"3. Answer the EXACT question asked. Use ALL sources that have relevant content."
                f"{_conflict_note}{_cond_note}{_fallback_note}\n\n"
                f"{summary_section}CONTEXT:\n{context}\n\nQUESTION: {req.query}\nANSWER:"
            )
            answer_llm = llm or get_eval_llm()
            response = answer_llm.invoke(citation_prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            if not answer.strip():
                answer = "Not mentioned in documents."
            sources = _extract_cited_sources(answer, sources)
        except Exception as exc:
            traceback.print_exc()
            answer  = f"[LLM error: {exc}]"
            sources = list(seen_sources)
        llm_ms = (time.perf_counter() - t_llm_start) * 1000

    ragas_scores: Optional[RagasScores] = None
    ragas_per_chunk: list[dict[str, Any]] = []
    ragas_ms = 0.0
    if req.run_ragas and req.generate_answer and chunks:
        t_ragas_start = time.perf_counter()
        try:
            ragas_scores, ragas_per_chunk = _run_ragas(req.query, answer or "", chunks)
        except RuntimeError as _ragas_err:
            logger.warning("[RAGAS] skipped — no LLM configured: %s", _ragas_err)
        ragas_ms = (time.perf_counter() - t_ragas_start) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000
    response = EvalResponse(
        query=req.query, doc_metadata=doc_metadata, chunk_metadata=chunk_metadata,
        chunks=chunks, total_chunks_in_store=vs.count(),
        ragas=ragas_scores, ragas_per_chunk=ragas_per_chunk,
        answer=answer, sources=sources, has_conflict=has_conflict,
        conflict_insurers=list(conflict_insurers) if conflict_insurers else [],
        timing={
            "retrieval_ms": round(retrieval_ms, 1), "llm_ms": round(llm_ms, 1),
            "ragas_ms": round(ragas_ms, 1), "total_ms": round(total_ms, 1),
        },
    )
    try:
        if not (answer and answer.startswith("[LLM error")):
            kv.put(cache_key, response.model_dump(), query_embedding=_query_emb, query_text=req.query)
    except Exception as exc:
        logger.warning("[EVAL] KV cache store failed: %s", exc)
    return response


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING QUERY
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/eval/query/stream", summary="Stream answer tokens via SSE")
def eval_query_stream(req: EvalRequest):
    from fastapi.responses import StreamingResponse as SR

    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def generate():
        vs = get_vector_store()
        kv = get_kv_cache()
        t0 = time.perf_counter()

        current_sources = vs.list_sources()
        cache_key = kv.make_key(
            req.query, req.top_k, req.use_hybrid, req.use_reranker,
            req.generate_answer, req.run_ragas, current_sources,
        )
        cached = kv.get(cache_key)
        _q_emb = None
        if cached is None:
            try:
                _q_emb = get_compressor()._model.encode(
                    [req.query], normalize_embeddings=True, show_progress_bar=False
                )[0]
                sem_hit = kv.semantic_get(_q_emb)
                if sem_hit is not None and req.generate_answer and not sem_hit.get("answer"):
                    sem_hit = None
                cached = sem_hit
            except Exception:
                pass

        if cached is not None:
            cached["doc_metadata"]["from_cache"] = True
            cached_answer = cached.get("answer") or ""
            cached["answer"] = ""
            yield _sse({"type": "retrieval_complete", **cached})
            for i in range(0, len(cached_answer), 6):
                yield _sse({"type": "token", "text": cached_answer[i:i+6]})
            yield _sse({"type": "done", "timing_ms": round((time.perf_counter()-t0)*1000,1),
                        "llm_ms": 0, "from_cache": True})
            return

        if vs.count() == 0:
            yield _sse({"type": "error", "message": "Vector store is empty."})
            return

        ret              = _retrieve(req)
        raw_docs         = ret["raw_docs"]
        raw_docs_for_llm = ret["raw_docs_for_llm"]
        summary_context  = ret["summary_context"]
        filter_meta      = ret["filter_meta"]
        query_meta       = ret["query_meta"]
        retrieval_ms     = ret["retrieval_ms"]

        chunks_data = []
        seen_sources: set = set()
        sections_hit: list = []
        policy_types_hit: list = []
        sections_seen: set = set()
        policy_types_seen: set = set()

        for idx, doc in enumerate(raw_docs_for_llm):
            meta    = dict(doc.metadata)
            section = meta.get("section") or classify_chunk_intent(
                doc.page_content[:150], doc_type=meta.get("doc_type", "policy_document"), llm=None
            )
            sim    = float(meta.get("similarity", 0.0))
            rerank = float(meta["rerank_score"]) if "rerank_score" in meta else None
            src    = meta.get("source", "unknown")
            chunks_data.append({
                "chunk_index": idx, "text": doc.page_content,
                "char_count": len(doc.page_content), "word_count": len(doc.page_content.split()),
                "similarity_score": round(sim, 4),
                "rerank_score": round(rerank, 4) if rerank is not None else None,
                "retrieval_method": meta.get("retrieval_method", "dense"),
                "metadata": meta, "section": section,
            })
            seen_sources.add(src)
            if section not in sections_seen:
                sections_seen.add(section); sections_hit.append(section)
            pt = meta.get("policy_type", "general")
            if pt not in policy_types_seen:
                policy_types_seen.add(pt); policy_types_hit.append(pt)

        has_conflict_local, conflict_ins = detect_conflict(raw_docs)
        chunk_metadata = [
            {
                "chunk_index": c["chunk_index"], "source": c["metadata"].get("source", "unknown"),
                "page": c["metadata"].get("page"), "insurer": c["metadata"].get("insurer", "UNKNOWN"),
                "policy_type": c["metadata"].get("policy_type", "general"),
                "doc_type": c["metadata"].get("doc_type", "policy_document"),
                "section": c["section"], "similarity_score": c["similarity_score"],
                "rerank_score": c["rerank_score"], "retrieval_method": c["retrieval_method"],
                "char_count": c["char_count"], "word_count": c["word_count"],
                "compressed": c["metadata"].get("compressed", False),
            }
            for c in chunks_data
        ]
        doc_metadata = {
            "query_insurer_hint":      query_meta.get("insurer"),
            "query_policy_type_hint":  query_meta.get("policy_type"),
            "metadata_filter_applied": filter_meta,
            "sources_retrieved":       list(seen_sources),
            "sections_hit":            sections_hit,
            "policy_types_hit":        policy_types_hit,
            "retrieval_method":        raw_docs[0].metadata.get("retrieval_method", "dense") if raw_docs else "n/a",
            "total_chunks_in_store":   vs.count(),
            "chunks_retrieved":        len(chunks_data),
            "from_cache":              False,
        }
        yield _sse({
            "type": "retrieval_complete", "query": req.query,
            "doc_metadata": doc_metadata, "chunk_metadata": chunk_metadata,
            "chunks": chunks_data, "total_chunks_in_store": vs.count(),
            "ragas": None, "ragas_per_chunk": [], "answer": "",
            "sources": list(seen_sources),
            "has_conflict": has_conflict_local,
            "conflict_insurers": list(conflict_ins) if conflict_ins else [],
            "timing": {"retrieval_ms": retrieval_ms, "llm_ms": 0, "ragas_ms": 0, "total_ms": retrieval_ms},
        })

        if not req.generate_answer or not raw_docs:
            yield _sse({"type": "done", "timing_ms": round((time.perf_counter()-t0)*1000,1), "llm_ms": 0})
            return

        llm = get_eval_llm()
        if llm is None:
            yield _sse({"type": "error", "message": "LLM not configured."})
            return

        context = _build_structured_context(raw_docs_for_llm, max_chars=MAX_CONTEXT_CHARS)
        summary_section = f"SUMMARIES:\n{summary_context}\n\n" if summary_context else ""
        conflict_hint = (
            f"\nNOTE: Multiple insurers ({', '.join(sorted(conflict_ins))}). Keep facts separate."
            if has_conflict_local else ""
        )
        stream_unique_srcs = list(dict.fromkeys(
            d.metadata.get("source", "") for d in raw_docs_for_llm if d.metadata.get("source")
        ))
        stream_src_list = ", ".join(f'"{s}"' for s in stream_unique_srcs)
        stream_ctx_covered = _context_covers_query(req.query, raw_docs_for_llm)
        stream_fallback_note = (
            "\nIMPORTANT: The retrieved document chunks do NOT contain explicit information "
            "about this specific topic. Provide a helpful, accurate general insurance "
            "explanation from your training knowledge and begin your answer with: "
            "'General knowledge (not from uploaded documents): '"
            if not stream_ctx_covered else ""
        )
        prompt = (
            f"You are an insurance document assistant. "
            f"The CONTEXT below contains chunks from these sources: {stream_src_list}.\n"
            f"RULES:\n"
            f"1. If the CONTEXT contains relevant information: answer using ONLY the CONTEXT "
            f"and cite every fact as [Doc: filename, p.N].\n"
            f"2. If the CONTEXT does NOT explicitly cover the question: provide a clear, "
            f"accurate general insurance explanation from your training knowledge and label "
            f"it 'General knowledge (not from uploaded documents): '.\n"
            f"3. Answer the EXACT question asked. Use ALL sources that have relevant content."
            f"{conflict_hint}{stream_fallback_note}\n\n"
            f"{summary_section}CONTEXT:\n{context}\n\nQUESTION: {req.query}\nANSWER:"
        )
        t_llm = time.perf_counter()
        full_answer = ""
        try:
            for chunk in llm.stream(prompt):
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    full_answer += token
                    yield _sse({"type": "token", "text": token})
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
            return

        llm_ms   = round((time.perf_counter() - t_llm) * 1000, 1)
        total_ms = round((time.perf_counter() - t0) * 1000, 1)
        cited_sources = _extract_cited_sources(full_answer, list(seen_sources))
        yield _sse({"type": "done", "timing_ms": total_ms, "llm_ms": llm_ms,
                    "retrieval_ms": retrieval_ms, "sources": cited_sources})
        try:
            if full_answer and not full_answer.startswith("[LLM error"):
                full_resp = {
                    "query": req.query, "doc_metadata": doc_metadata,
                    "chunk_metadata": chunk_metadata, "chunks": chunks_data,
                    "total_chunks_in_store": vs.count(), "ragas": None,
                    "ragas_per_chunk": [], "answer": full_answer,
                    "sources": cited_sources, "has_conflict": has_conflict_local,
                    "conflict_insurers": list(conflict_ins) if conflict_ins else [],
                    "timing": {"retrieval_ms": retrieval_ms, "llm_ms": llm_ms, "ragas_ms": 0, "total_ms": total_ms},
                }
                kv.put(cache_key, full_resp, query_embedding=_q_emb, query_text=req.query)
        except Exception:
            pass

    return SR(generate(), media_type="text/event-stream",
              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ══════════════════════════════════════════════════════════════════════════════
# RAGAS (parallel, 3 LLM calls at once)
# ══════════════════════════════════════════════════════════════════════════════

def _clean_json(raw: str) -> str:
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', raw)
    return match.group(1).strip() if match else raw

def _llm_invoke_json(llm, system_content: str, user_content: str) -> str:
    from langchain_core.messages import SystemMessage, HumanMessage
    resp = llm.invoke([SystemMessage(content=system_content), HumanMessage(content=user_content)])
    return resp.content if hasattr(resp, "content") else str(resp)

def _run_ragas(query: str, answer: str, chunks: list[ChunkInfo]) -> tuple[RagasScores, list[dict]]:
    llm = get_insurance_llm(temperature=0, max_tokens=80)
    SYSTEM_JSON = "Output ONLY valid JSON. No prose, no markdown."
    chunk_lines  = "\n".join(f"[Chunk {c.chunk_index}] {c.text[:400]}" for c in chunks)
    context_blob = "\n\n---\n\n".join(f"[Chunk {c.chunk_index}]\n{c.text[:400]}" for c in chunks)

    tasks = {
        "per_chunk": (
            f'Score each chunk\'s relevance to the query from 0.0 to 1.0.\n\n'
            f'Query: "{query}"\n\nChunks:\n{chunk_lines}\n\n'
            f'Return: [{{"chunk_index": 0, "relevance": 0.85, "reason": "brief"}}]'
        ),
        "faithfulness": (
            f'Score from 0.0 to 1.0.\n\nQuery: "{query}"\n\nContext:\n{context_blob}\n\nAnswer:\n{answer[:800]}\n\n'
            f'- faithfulness: fraction of answer claims supported by context\n'
            f'- answer_relevancy: how well the answer addresses the query\n\n'
            f'Return ONLY: {{"faithfulness": 0.9, "answer_relevancy": 0.85}}'
        ),
        "precision": (
            f'Score from 0.0 to 1.0.\n\nQuery: "{query}"\n\nContext:\n{context_blob}\n\n'
            f'- context_precision: fraction of retrieved chunks that are relevant\n'
            f'- context_recall: does the context contain enough to fully answer\n\n'
            f'Return ONLY: {{"context_precision": 0.9, "context_recall": 0.85}}'
        ),
    }
    raw_results: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        future_to_name = {
            pool.submit(_llm_invoke_json, llm, SYSTEM_JSON, prompt): name
            for name, prompt in tasks.items()
        }
        for future in as_completed(future_to_name):
            name = future_to_name[future]
            try:
                raw_results[name] = future.result()
            except Exception as exc:
                logger.warning("[RAGAS] %s failed: %s", name, exc)

    ragas_per_chunk: list[dict] = []
    try:
        ragas_per_chunk = json.loads(_clean_json(raw_results.get("per_chunk", "[]")))
    except Exception:
        ragas_per_chunk = [
            {"chunk_index": c.chunk_index, "relevance": c.similarity_score, "reason": "fallback"}
            for c in chunks
        ]
    avg = sum(r.get("relevance", 0) for r in ragas_per_chunk) / (len(ragas_per_chunk) or 1)

    scores: dict[str, float] = {}
    is_fallback = False
    for name in ("faithfulness", "precision"):
        raw = raw_results.get(name, "")
        if not raw:
            is_fallback = True; continue
        try:
            scores.update(json.loads(_clean_json(raw)))
        except Exception:
            is_fallback = True

    def _safe(key: str) -> float:
        val = scores.get(key)
        if val is None: return round(avg, 3)
        try: return round(float(val), 3)
        except: return round(avg, 3)

    if is_fallback or not scores:
        return RagasScores(
            faithfulness=round(avg,3), answer_relevancy=round(avg,3),
            context_precision=round(avg,3), context_recall=round(avg,3),
            judge_model="fallback", is_fallback=True,
        ), ragas_per_chunk

    return RagasScores(
        faithfulness=_safe("faithfulness"), answer_relevancy=_safe("answer_relevancy"),
        context_precision=_safe("context_precision"), context_recall=_safe("context_recall"),
        judge_model=get_active_model_info().get("model") or "unknown", is_fallback=False,
    ), ragas_per_chunk


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARIES, CACHE, LLM, HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/eval/summaries")
def eval_summaries():
    ss = get_summary_store()
    summaries = ss.list_all()
    return {
        "count": len(summaries),
        "summaries": [
            {
                "source": s["metadata"].get("source", ""), "title": s["metadata"].get("title", ""),
                "summary_type": s["metadata"].get("summary_type", "document"),
                "doc_type": s["metadata"].get("doc_type", ""),
                "insurer": s["metadata"].get("insurer", "UNKNOWN"),
                "policy_type": s["metadata"].get("policy_type", "general"),
                "chunk_count": s["metadata"].get("chunk_count", 0),
                "source_type": s["metadata"].get("source_type", ""),
                "ingested_at": s["metadata"].get("ingested_at", ""), "text": s["text"],
            }
            for s in summaries
        ],
    }

class EvalRequestSimple(BaseModel):
    query: str
    top_k: int = 8

@app.post("/eval/summaries/search")
def eval_search_summaries(req: EvalRequestSimple):
    docs = get_summary_store().search(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "results": [
            {
                "source": d.metadata.get("source", ""), "title": d.metadata.get("title", ""),
                "summary_type": d.metadata.get("summary_type", "document"),
                "insurer": d.metadata.get("insurer", "UNKNOWN"),
                "policy_type": d.metadata.get("policy_type", "general"),
                "similarity": round(float(d.metadata.get("similarity", 0)), 4),
                "text": d.page_content,
            }
            for d in docs
        ],
    }

@app.get("/eval/cache/stats")
def eval_cache_stats():
    return get_kv_cache().stats()

@app.post("/eval/cache/flush")
def eval_cache_flush():
    return {"status": "ok", "entries_removed": get_kv_cache().flush()}

@app.post("/eval/cache/clear")
def eval_cache_clear():
    get_kv_cache().clear()
    return {"status": "ok"}

@app.get("/eval/llm-models")
def eval_llm_models():
    models = list_vllm_models()
    info   = get_active_model_info()
    return {"available_models": models, "active_model": info.get("model"), "backend": info.get("backend")}

class ModelSelectRequest(BaseModel):
    model: str

@app.post("/eval/llm-models/select")
def eval_llm_model_select(req: ModelSelectRequest):
    if not req.model.strip():
        return JSONResponse(status_code=400, content={"error": "model name cannot be empty"})
    global _llm_instance
    set_model_override(req.model.strip())
    _llm_instance = None
    get_eval_llm()
    return {"status": "ok", "active_model": req.model.strip()}

@app.get("/eval/llm-test")
def eval_llm_test():
    info = get_active_model_info()
    if info.get("backend") == "none":
        return JSONResponse(status_code=200, content={
            "status": "unconfigured", "backend": "none", "model": None,
            "message": "No LLM configured. Set VLLM_HOST, OPENAI_API_KEY, or ANTHROPIC_API_KEY.",
        })
    try:
        llm  = get_eval_llm()
        resp = llm.invoke("Reply with the single word: OK")
        text = resp.content if hasattr(resp, "content") else str(resp)
        return {"status": "ok", "backend": info.get("backend"), "model": info.get("model"), "reply": text.strip()[:80]}
    except Exception as exc:
        return JSONResponse(status_code=200, content={
            "status": "error", "backend": info.get("backend"), "model": info.get("model"), "message": str(exc),
        })

@app.get("/eval/health")
def health():
    try:
        vs    = get_vector_store()
        count = vs.count()
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})
    llm_info = get_active_model_info()
    return {
        "status": "ok", "chunks_in_store": count,
        "summaries_in_store": get_summary_store().count(),
        "cache_stats": get_kv_cache().stats(),
        "llm_backend": llm_info.get("backend", "none"),
        "llm_model": llm_info.get("model") or "not configured",
        "vllm_host": os.getenv("VLLM_HOST", "not-set"),
    }


@app.get("/", include_in_schema=False)
def serve_frontend():
    from fastapi.responses import FileResponse
    html_path = os.path.join(_here, "eval_frontend.html")
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return JSONResponse(status_code=404, content={"error": "eval_frontend.html not found"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("eval_api:app", host="0.0.0.0", port=8002, reload=False)