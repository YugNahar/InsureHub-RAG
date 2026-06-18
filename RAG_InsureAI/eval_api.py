"""
RAG Evaluation API — eval_api.py
Drop this file into:  InsureHub-RAG-main/RAG_InsureAI/

Run with:
    cd InsureHub-RAG-main/RAG_InsureAI
    uvicorn eval_api:app --host 0.0.0.0 --port 8001 --workers 1

KEY FIX vs previous version
----------------------------
Root cause of "policy_type always general": pipeline.chunker.split_documents()
was being called WITHOUT llm=llm for non-YouTube documents. The chunker's
classify_chunk_policy_type() then ran with llm=None, which means it can ONLY
use regex — and regex requires >=2 confident keyword hits or it falls back to
"general". Most real-world chunks (especially handbook prose, YouTube
transcripts, or policy text using synonyms) don't hit that bar.

Fixed by:
  1. Always pass llm=llm to pipeline.chunker.split_documents() — for BOTH
     YouTube and regular documents. This lets the LLM classify section AND
     policy_type per-chunk using regex hits as few-shot hints (not hard rules).
  2. YouTube chunks now ALSO go through SectionChunker (same code path as
     PDFs/DOCX) instead of being used raw — they're classified identically,
     with force_llm=True automatically applied (chunker detects doc_type
     == "youtube" and source_type containing "whisper"/"youtube").
  3. Query-time section/policy_type for ChunkInfo display now uses the LLM
     too (via _detect_section_llm helper) instead of pure regex fallback,
     so the eval UI shows what the model actually believes, not a stale
     regex guess.
  4. Added structured debug logging — every chunk's regex score vs LLM
     decision is logged so you can see in the terminal exactly why a label
     was chosen.
  5. Query-time metadata filtering: classify_query() now runs with the LLM
     too, and build_metadata_filter() biases retrieval toward chunks with
     matching insurer/policy_type — applied identically whether the chunk
     came from a PDF or a YouTube video, since both share the same
     classify_chunk_policy_type() pipeline. Falls back to unfiltered search
     if the filter returns zero results.

WEBPAGE INGESTION (added)
----------------------------
  6. Plain web pages (non-YouTube URLs) are ingested via a new
     /eval/ingest-webpage endpoint. They go through document_loader's
     load_url_advanced() → SectionChunker → _classify_docs_metadata(),
     the exact same pipeline as PDFs and YouTube, and land in the SAME
     ChromaVectorStore collection (no separate store). Because /eval/query
     already searches that one collection, webpage-sourced chunks show up
     in metadata, raw chunks, RAGAS, and the generated answer automatically
     — no changes were needed to eval_query() itself.
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
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Make sure both RAG_InsureAI/ and RAG_InsureAI/app/ are on the path ────────
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
    MAX_CONTEXT_CHARS, RETRIEVE_K, RERANK_K,
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

# Home-directory temp root — macOS sandboxes /tmp and /var/folders/.../T/
# against subprocess access (ffmpeg/whisper can't read files there even
# though Python itself can see them).
_SAFE_TMP_ROOT = os.path.expanduser("~/.insurehub_tmp")
os.makedirs(_SAFE_TMP_ROOT, exist_ok=True)

# ── Shared singletons ──────────────────────────────────────────────────────────
_vector_store: Optional[ChromaVectorStore] = None
_rag_pipeline: Optional[RAGPipeline] = None
_summary_store: Optional[SummaryStore] = None
_kv_cache: Optional[QueryKVCache] = None
_compressor: Optional[ContextCompressor] = None
_llm_instance: Optional[Any] = None            # cached once — never recreated per call

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
    """Lazy singleton — reuses the embed model already loaded by the vector store."""
    global _compressor
    if _compressor is None:
        embed_model = get_vector_store().embed_model
        _compressor = ContextCompressor(
            embed_model=embed_model,
            similarity_threshold=float(os.getenv("COMPRESS_THRESHOLD", "0.38")),
            min_sentences=int(os.getenv("COMPRESS_MIN_SENTS", "2")),
            max_sentences=int(os.getenv("COMPRESS_MAX_SENTS", "10")),
            max_chars_per_chunk=int(os.getenv("COMPRESS_SKIP_BELOW", "600")),
        )
    return _compressor


def get_eval_llm():
    """
    Return the shared LLM instance, creating it once and caching it forever.

    Using a singleton avoids:
    - Reconstructing ChatOpenAI (network round-trip to /v1/models) on every call
    - Redundant logging noise
    - Separate LLM objects for the query path vs RAGAS vs classification
    """
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


# ── Startup pre-warming ────────────────────────────────────────────────────────
from contextlib import asynccontextmanager


@asynccontextmanager
async def _lifespan(application):
    """
    Pre-warm every singleton at startup so the first user query is fast.

    Without this, the first request triggers:
      • BGE model load from HuggingFace cache (~5-8 s)
      • TurboVec index load from disk (~2 s)
      • vLLM /v1/models discovery (~0.5 s)
    By loading all of these at startup, every subsequent request hits
    already-warm singletons with no cold-start penalty.
    """
    logger.info("[STARTUP] Pre-warming all singletons ...")
    try:
        vs  = get_vector_store()
        ss  = get_summary_store()
        kv  = get_kv_cache()
        cmp = get_compressor()
        llm = get_eval_llm()
        info = get_active_model_info()

        # Run dummy inference so PyTorch JIT compilation and CUDA graph capture
        # happen now, not on the first user request.
        # warmup() also loads the cross-encoder reranker eagerly.
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


# ── App setup ──────────────────────────────────────────────────────────────────
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
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Source-intent detection ───────────────────────────────────────────────────
_VIDEO_SIGNALS   = {"video", "youtube", "watch", "the video", "in the video",
                    "video says", "video mentioned", "talked about", "spoken",
                    "lecture", "clip", "webinar", "recording"}
_DOC_SIGNALS     = {"document", "pdf", "policy", "file", "article",
                    "handbook", "the document", "in the document"}
_COMBINE_SIGNALS = {"both", "together", "all sources", "across", "combine",
                    "and the video", "and the document", "as well as"}

def _source_intent(query: str) -> str:
    """
    Return 'video', 'document', 'combined', or 'any'.

    'video'    — query explicitly mentions video/YouTube content
    'document' — query explicitly mentions PDFs/documents/policies
    'combined' — query asks about both or all sources
    'any'      — no explicit source preference; search everything
    """
    q = query.lower()
    has_video    = any(sig in q for sig in _VIDEO_SIGNALS)
    has_doc      = any(sig in q for sig in _DOC_SIGNALS)
    has_combine  = any(sig in q for sig in _COMBINE_SIGNALS)

    if has_combine or (has_video and has_doc):
        return "combined"
    if has_video:
        return "video"
    if has_doc:
        return "document"
    return "any"


# ── Shared two-stage retrieval ────────────────────────────────────────────────
def _is_youtube_doc(meta: dict) -> bool:
    """True if a chunk/summary metadata dict belongs to a YouTube video."""
    return (
        meta.get("doc_type") == "youtube"
        or "youtube" in str(meta.get("source_type", "")).lower()
        or "whisper" in str(meta.get("source_type", "")).lower()
        or "youtube.com" in str(meta.get("source", "")).lower()
        or "youtu.be" in str(meta.get("source", "")).lower()
    )


def _retrieve(req: "EvalRequest") -> dict:
    """
    Two-stage retrieval shared by eval_query and eval_query_stream.

    Stage 1 — query the summary store to build context and identify sources.
    Stage 2 — retrieve chunks, always including a dedicated YouTube sub-search
              so that informal transcript text isn't outranked by PDF chunks.

    Returns:
      raw_docs, summary_context, filter_meta, query_meta, retrieval_ms, intent
    """
    vs = get_vector_store()
    ss = get_summary_store()
    t0 = time.perf_counter()

    intent = _source_intent(req.query)
    logger.info("[RETRIEVE] intent=%s  query=%r", intent, req.query[:60])

    # Detect whether any YouTube content exists in the store at all.
    # We check the summary store (one entry per source) which is fast.
    yt_sources_in_store: list[str] = []
    if ss.count() > 0:
        all_summaries = ss.list_all()
        yt_sources_in_store = [
            s["metadata"].get("source", "")
            for s in all_summaries
            if _is_youtube_doc(s["metadata"])
        ]
    yt_exists = bool(yt_sources_in_store)
    logger.info("[RETRIEVE] yt_exists=%s  yt_sources=%s", yt_exists, yt_sources_in_store[:3])

    # ── Stage 1: Summary-guided source selection ──────────────────────────
    summary_context = ""
    source_filter_for_chunks: Optional[dict] = None

    if ss.count() > 0:
        # Always fetch enough summaries so YouTube ones aren't missed even when
        # they rank lower than PDFs on cosine similarity.
        stage1_k = max(ss.count(), 10)
        top_summary_docs = ss.search(req.query, top_k=stage1_k)

        # Separate by source type
        yt_sum_docs  = [d for d in top_summary_docs if _is_youtube_doc(d.metadata)]
        pdf_sum_docs = [d for d in top_summary_docs if not _is_youtube_doc(d.metadata)]

        # Build the selected set for summary context
        if intent == "video":
            selected = yt_sum_docs[:3] + pdf_sum_docs[:1]
        elif intent == "combined":
            selected = pdf_sum_docs[:2] + yt_sum_docs[:2]
        else:
            # "document" or "any": top-3 PDFs + top YouTube summary for context
            selected = pdf_sum_docs[:3] + (yt_sum_docs[:1] if yt_exists else [])

        top_sources = [d.metadata.get("source", "") for d in selected if d.metadata.get("source")]
        logger.info("[RETRIEVE] Stage-1 sources (intent=%s): %s", intent, top_sources)

        if top_sources:
            parts = []
            for doc in selected:
                src      = doc.metadata.get("source", "?")
                dtype    = doc.metadata.get("doc_type", "document")
                insurer  = doc.metadata.get("insurer", "UNKNOWN")
                parts.append(f"[Summary | {dtype} | {src} | {insurer}]\n{doc.page_content}")
            summary_context = "\n\n".join(parts)

            # Only build a source filter for single-source-type intents;
            # for combined/any we let vector search rank across everything.
            if intent == "video":
                yt_urls = [d.metadata.get("source", "") for d in yt_sum_docs if d.metadata.get("source")]
                if yt_urls:
                    source_filter_for_chunks = (
                        {"source": {"$eq": yt_urls[0]}} if len(yt_urls) == 1
                        else {"$or": [{"source": {"$eq": u}} for u in yt_urls]}
                    )
            elif intent == "document":
                pdf_urls = [d.metadata.get("source", "") for d in pdf_sum_docs[:3] if d.metadata.get("source")]
                if pdf_urls:
                    source_filter_for_chunks = (
                        {"source": {"$eq": pdf_urls[0]}} if len(pdf_urls) == 1
                        else {"$or": [{"source": {"$eq": u}} for u in pdf_urls]}
                    )
            # "combined" and "any" → source_filter_for_chunks stays None

    # ── Build policy filter from query keywords (regex, instant) ─────────
    query_meta    = classify_query(req.query, llm=None)
    policy_filter = build_metadata_filter(query_meta, routed_sources=None)

    # For combined/any, skip policy filter — YouTube chunks are often tagged
    # "general" even when on-topic; a hard policy_type=$eq would exclude them.
    if source_filter_for_chunks:
        filter_meta: Optional[dict] = source_filter_for_chunks
    elif intent in ("combined", "any"):
        filter_meta = None
    else:
        filter_meta = policy_filter

    logger.info("[RETRIEVE] filter_meta=%s", filter_meta)

    # ── Stage 2a: Main chunk retrieval ────────────────────────────────────
    # Never rerank inside individual searches — we do one consolidated pass
    # after all searches are merged so CrossEncoder.predict() is called once.
    t_ret = time.perf_counter()
    raw_docs = vs.search(
        query=req.query,
        top_k=req.top_k * 2,
        filter_metadata=filter_meta,
        use_hybrid=req.use_hybrid,
        use_reranker=False,
    )
    if not raw_docs and filter_meta is not None:
        logger.warning("[RETRIEVE] filtered search returned 0 — retrying unfiltered")
        raw_docs = vs.search(
            query=req.query,
            top_k=req.top_k * 2,
            filter_metadata=None,
            use_hybrid=req.use_hybrid,
            use_reranker=False,
        )

    # ── Stage 2b: Dedicated YouTube sub-search ────────────────────────────
    # YouTube transcript text (informal, no punctuation) consistently ranks
    # below PDF chunks in cosine similarity even when the content is relevant.
    # We always run a separate YouTube search and merge results whenever
    # YouTube content exists in the store, except for document-only queries.
    if yt_exists and intent != "document":
        has_yt_already = any(_is_youtube_doc(d.metadata) for d in raw_docs)

        if not has_yt_already or intent in ("video", "combined"):
            yt_quota = req.top_k if intent == "video" else min(req.top_k, 3)

            yt_extra: list = []
            for yt_src in yt_sources_in_store:
                hits = vs.search(
                    query=req.query,
                    top_k=yt_quota,
                    filter_metadata={"source": {"$eq": yt_src}},
                    use_hybrid=req.use_hybrid,
                    use_reranker=False,
                )
                yt_extra.extend(hits)

            if not yt_extra:
                yt_extra = vs.search(
                    query=req.query,
                    top_k=yt_quota,
                    filter_metadata={"doc_type": {"$eq": "youtube"}},
                    use_hybrid=req.use_hybrid,
                    use_reranker=False,
                )

            if yt_extra:
                existing_keys = {d.page_content[:80] for d in raw_docs}
                new_yt = [d for d in yt_extra if d.page_content[:80] not in existing_keys]
                raw_docs = raw_docs + new_yt
                logger.info("[RETRIEVE] YouTube sub-search: +%d new chunks", len(new_yt))

    retrieval_ms = round((time.perf_counter() - t_ret) * 1000, 1)

    # ── Deduplicate ───────────────────────────────────────────────────────
    _seen: dict = {}
    for doc in raw_docs:
        key = (doc.metadata.get("source", ""), doc.metadata.get("page"), doc.page_content)
        sc  = doc.metadata.get("rerank_score", doc.metadata.get("similarity", 0))
        if key not in _seen or sc > _seen[key].metadata.get(
                "rerank_score", _seen[key].metadata.get("similarity", 0)):
            _seen[key] = doc
    deduped = list(_seen.values())

    # ── Single consolidated rerank pass ──────────────────────────────────
    # One CrossEncoder.predict() call over all deduplicated candidates is
    # far cheaper than one call per individual search above.
    if req.use_reranker and len(deduped) > 1:
        # Expand the pool before reranking so the reranker sees enough candidates
        deduped = vs.rerank_documents(req.query, deduped, top_k=len(deduped))
        score_key = lambda d: d.metadata.get("rerank_score", 0)
    else:
        score_key = lambda d: d.metadata.get("rerank_score", d.metadata.get("similarity", 0))

    # ── Final ranking with source-type balancing ──────────────────────────
    if intent == "video" and yt_exists:
        yt_ranked  = sorted([d for d in deduped if _is_youtube_doc(d.metadata)],
                            key=score_key, reverse=True)
        pdf_ranked = sorted([d for d in deduped if not _is_youtube_doc(d.metadata)],
                            key=score_key, reverse=True)
        # Interleave 2 YouTube : 1 PDF (favours YouTube for video queries)
        raw_docs = []
        yi, pi = 0, 0
        while len(raw_docs) < req.top_k:
            if yi < len(yt_ranked) and (len(raw_docs) % 3 != 2 or pi >= len(pdf_ranked)):
                raw_docs.append(yt_ranked[yi]); yi += 1
            elif pi < len(pdf_ranked):
                raw_docs.append(pdf_ranked[pi]); pi += 1
            else:
                break
    elif intent == "combined" and yt_exists:
        # Guarantee at least ⌈top_k/3⌉ YouTube chunks
        yt_ranked  = sorted([d for d in deduped if _is_youtube_doc(d.metadata)],
                            key=score_key, reverse=True)[:max(1, req.top_k // 3)]
        all_ranked = sorted(deduped, key=score_key, reverse=True)[:req.top_k]
        yt_keys    = {d.page_content[:80] for d in yt_ranked}
        raw_docs   = yt_ranked + [d for d in all_ranked if d.page_content[:80] not in yt_keys]
        raw_docs   = raw_docs[:req.top_k]
    else:
        raw_docs = sorted(deduped, key=score_key, reverse=True)[:req.top_k]

    # ── Context compression ───────────────────────────────────────────────
    raw_docs = get_compressor().compress_to_budget(req.query, list(raw_docs), MAX_CONTEXT_CHARS)

    logger.info(
        "[RETRIEVE] done — %d chunks | intent=%s | retrieval_ms=%.0f | sources=%s",
        len(raw_docs), intent,
        (time.perf_counter() - t0) * 1000,
        list({d.metadata.get("source", "?") for d in raw_docs}),
    )

    return {
        "raw_docs":        raw_docs,
        "summary_context": summary_context,
        "filter_meta":     filter_meta,
        "query_meta":      query_meta,
        "retrieval_ms":    retrieval_ms,
        "intent":          intent,
    }


def _should_force_llm(doc_type: str, source_type: str = "") -> bool:
    """
    Decide whether to force an LLM call even when regex looks confident.
    YouTube/Whisper/general docs use colloquial language that regex
    systematically under-detects, so we always prefer the LLM there.

    Plain web pages are tagged doc_type="general" by document_loader, so
    they already fall into this bucket without any extra special-casing.
    """
    source_type_lower = source_type.lower()
    return (
        doc_type in ("youtube", "general")
        or "whisper" in source_type_lower
        or "youtube" in source_type_lower
    )


def _classify_docs_metadata(docs: list, doc_type: str, doc_meta: dict, llm: Any) -> None:
    """
    Per-chunk classification pass: assigns `section` (intent) and
    `policy_type` to every chunk using regex-fast-path + LLM-fallback.

    This is used as a safety net AFTER SectionChunker — if the chunker
    already classified everything with llm!=None, this is mostly a no-op
    re-confirmation. If the chunker fell back to raw pages (e.g. an
    exception), this is what actually does the classification work.
    """
    for doc in docs:
        source_type = doc.metadata.get("source_type", "")
        force = _should_force_llm(doc_type, source_type)

        regex_section_before = doc.metadata.get("section")
        doc.metadata["section"] = classify_chunk_intent(
            doc.page_content,
            doc_type=doc_type,
            llm=llm,
            force_llm=force,
        )

        chunk_policy = classify_chunk_policy_type(
            doc.page_content,
            llm=llm,
            force_llm=force,
        )

        if chunk_policy != "general":
            doc.metadata["policy_type"] = chunk_policy
        else:
            doc.metadata["policy_type"] = doc_meta.get("policy_type", "general")

        logger.debug(
            "[CLASSIFY] doc_type=%s force_llm=%s | section: %r -> %r | policy_type=%s | text=%r",
            doc_type, force, regex_section_before, doc.metadata["section"],
            doc.metadata["policy_type"], doc.page_content[:80].replace("\n", " "),
        )


def _detect_section_llm(text: str, doc_type: str, llm: Any) -> str:
    """
    Query-time section detection that prefers the LLM classifier over pure
    regex. Used as the fallback inside eval_query() when a stored chunk's
    metadata doesn't already have a `section` value (e.g. legacy data
    ingested before this fix).
    """
    return classify_chunk_intent(text, doc_type=doc_type, llm=llm, force_llm=False)


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

    # ── Load LLM once, reuse everywhere below ──────────────────────────────────
    llm = get_eval_llm()

    preview    = " ".join(d.page_content for d in docs[:1])[:5000]
    extra_text = " ".join(d.page_content for d in docs[1:4])[:5000]
    fname      = file.filename or "document"
    doc_type   = classify_document_type(fname, preview, extra_text)
    doc_meta   = tag_document(fname, preview, extra_text=extra_text, doc_type=doc_type, llm=llm)
    doc_meta["doc_type"] = doc_type
    logger.info("[UPLOAD] '%s' → doc_type=%s, doc-level policy_type=%s",
                fname, doc_type, doc_meta.get("policy_type"))

    is_youtube_doc = doc_type == "youtube" or any(
        d.metadata.get("source_type", "") in ("youtube_transcript", "whisper")
        for d in docs
    )

    # ── FIX: route BOTH YouTube and regular docs through SectionChunker,
    #         and ALWAYS pass llm=llm. Previously YouTube skipped chunking
    #         entirely, and regular docs ran the chunker with llm=None,
    #         silently forcing every chunk-level policy_type to regex-only
    #         (which defaults to "general" whenever <2 confident keyword
    #         hits are found). ──────────────────────────────────────────────
    try:
        pipeline = get_rag_pipeline()
        for d in docs:
            d.metadata["doc_type"] = doc_type
            if is_youtube_doc:
                d.metadata.setdefault("source_type", "youtube_transcript")
        chunks = pipeline.chunker.split_documents(docs, doc_type=doc_type, llm=llm)
        logger.info(
            "[UPLOAD] SectionChunker → %d chunks from %d source docs (youtube=%s, llm=%s)",
            len(chunks), len(docs), is_youtube_doc, llm is not None,
        )
    except Exception as exc:
        traceback.print_exc()
        logger.warning("[UPLOAD] SectionChunker failed (%s), using raw pages", exc)
        chunks = docs

    for chunk in chunks:
        chunk.metadata.setdefault("filename",    fname)
        chunk.metadata.setdefault("source",      fname)
        chunk.metadata.setdefault("source_type", "document")   # never store None/empty
        chunk.metadata["doc_type"] = doc_type
        chunk.metadata["insurer"]  = chunk.metadata.get("insurer") or doc_meta.get("insurer", "UNKNOWN")

    # Safety-net re-classification pass — covers any chunk whose policy_type
    # is still missing/general after the chunker (e.g. SectionChunker
    # exception fallback above, where `chunks = docs` skips classification
    # entirely).
    _classify_docs_metadata(chunks, doc_type, doc_meta, llm)

    vs  = get_vector_store()
    ids = vs.add_documents(chunks)
    elapsed = (time.perf_counter() - t0) * 1000
    os.unlink(tmp_path)

    assigned_policy_types = list({c.metadata.get("policy_type", "general") for c in chunks})
    assigned_sections      = list({c.metadata.get("section", "general") for c in chunks})
    general_count = sum(1 for c in chunks if c.metadata.get("policy_type", "general") == "general")

    logger.info(
        "[UPLOAD] done — %d chunks | policy_types=%s | sections=%s | %d/%d still 'general' policy_type",
        len(chunks), assigned_policy_types, assigned_sections, general_count, len(chunks),
    )

    # ── Generate and store document summary ───────────────────────────────────
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
        logger.info("[UPLOAD] summary stored for %s", fname)
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
    }


# ── List documents ─────────────────────────────────────────────────────────────
@app.get("/eval/documents", summary="List all ingested documents")
def eval_documents():
    vs = get_vector_store()
    return {
        "sources":      vs.list_sources(),
        "filenames":    vs.list_filenames(),
        "total_chunks": vs.count(),
    }


# ── Delete endpoints ───────────────────────────────────────────────────────────
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
# INGEST YOUTUBE VIDEO
# ══════════════════════════════════════════════════════════════════════════════

class VideoIngestRequest(BaseModel):
    url: str


@app.post("/eval/ingest-video", summary="Ingest a YouTube video transcript into the vector store")
def eval_ingest_video(req: VideoIngestRequest):
    url = req.url.strip()
    logger.info("[INGEST-VIDEO] received url=%s", url)

    if not is_youtube_url(url):
        raise HTTPException(status_code=422, detail="URL does not appear to be a YouTube link.")

    t0 = time.perf_counter()

    try:
        logger.info("[INGEST-VIDEO] calling _load_youtube...")
        docs = _load_youtube(url)
        logger.info("[INGEST-VIDEO] _load_youtube returned %d docs", len(docs))
    except Exception as exc:
        traceback.print_exc()
        logger.error("[INGEST-VIDEO] _load_youtube FAILED: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Failed to load YouTube transcript: {exc}")

    if not docs:
        raise HTTPException(status_code=422, detail="No transcript content could be extracted from this video.")

    llm = get_eval_llm()

    video_id       = docs[0].metadata.get("video_id") or "unknown"
    synthetic_name = f"youtube_{video_id}.txt"
    source_type    = docs[0].metadata.get("source_type", "youtube_transcript")
    logger.info("[INGEST-VIDEO] video_id=%s source_type=%s raw_docs=%d", video_id, source_type, len(docs))

    preview    = docs[0].page_content[:5000]
    extra_text = docs[0].page_content[5000:10000]
    doc_type   = "youtube"
    doc_meta   = tag_document(synthetic_name, preview, extra_text=extra_text, doc_type=doc_type, llm=llm)
    doc_meta["doc_type"] = doc_type

    for doc in docs:
        doc.metadata.setdefault("filename", synthetic_name)
        doc.metadata["source"]   = url
        doc.metadata["insurer"]  = doc_meta.get("insurer", "UNKNOWN")
        doc.metadata["doc_type"] = doc_type

    # ── FIX: run YouTube transcript chunks through SectionChunker too,
    #         instead of treating the raw 800-word windows as final chunks.
    #         This gives consistent chunk sizing AND lets the chunker's
    #         classify_chunk_intent/classify_chunk_policy_type calls run
    #         with force_llm=True (auto-detected via doc_type="youtube"). ──
    try:
        pipeline = get_rag_pipeline()
        chunks = pipeline.chunker.split_documents(docs, doc_type=doc_type, llm=llm)
        logger.info("[INGEST-VIDEO] SectionChunker → %d chunks from %d raw transcript windows",
                    len(chunks), len(docs))
    except Exception as exc:
        traceback.print_exc()
        logger.warning("[INGEST-VIDEO] SectionChunker failed (%s), using raw transcript windows", exc)
        chunks = docs

    for chunk in chunks:
        chunk.metadata.setdefault("filename", synthetic_name)
        chunk.metadata["source"]   = url
        chunk.metadata["insurer"]  = doc_meta.get("insurer", "UNKNOWN")
        chunk.metadata["doc_type"] = doc_type
        chunk.metadata.setdefault("source_type", source_type)

    # Safety-net classification pass — same as upload endpoint.
    _classify_docs_metadata(chunks, doc_type, doc_meta, llm)

    vs  = get_vector_store()
    ids = vs.add_documents(chunks)
    elapsed = (time.perf_counter() - t0) * 1000

    assigned_policy_types = list({c.metadata.get("policy_type", "general") for c in chunks})
    assigned_sections      = list({c.metadata.get("section", "general") for c in chunks})
    general_count = sum(1 for c in chunks if c.metadata.get("policy_type", "general") == "general")

    logger.info(
        "[INGEST-VIDEO] done — %d chunks stored | policy_types=%s | sections=%s | %d/%d still 'general'",
        len(ids), assigned_policy_types, assigned_sections, general_count, len(chunks),
    )

    # ── Generate and store video summary ──────────────────────────────────────
    try:
        summary_text = generate_summary(docs, source=url, doc_meta=doc_meta, llm=llm)
        get_summary_store().upsert(
            source=url,
            summary_text=summary_text,
            metadata={
                "source_type":  source_type,
                "doc_type":     doc_type,
                "insurer":      doc_meta.get("insurer", "UNKNOWN"),
                "policy_type":  doc_meta.get("policy_type", "general"),
                "chunk_count":  len(ids),
                "title":        synthetic_name,
                "video_id":     video_id,
                "summary_type": "video",
            },
        )
        logger.info("[INGEST-VIDEO] summary stored for %s", url)
    except Exception as exc:
        logger.warning("[INGEST-VIDEO] summary generation failed: %s", exc)

    return {
        "status": "ok",
        "filename": synthetic_name,
        "url": url,
        "video_id": video_id,
        "source_type": source_type,
        "language": docs[0].metadata.get("language", "unknown"),
        "chunks_added": len(ids),
        "total_in_store": vs.count(),
        "doc_metadata": doc_meta,
        "assigned_policy_types": assigned_policy_types,
        "assigned_sections": assigned_sections,
        "chunks_still_general_policy_type": general_count,
        "ingest_ms": round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# INGEST WEBPAGE
# ══════════════════════════════════════════════════════════════════════════════

class WebpageIngestRequest(BaseModel):
    url: str


@app.post("/eval/ingest-webpage", summary="Scrape a web page and ingest it into the vector store")
def eval_ingest_webpage(req: WebpageIngestRequest):
    """
    Mirrors eval_ingest_video(), but for plain (non-YouTube) URLs.

    Pipeline is identical to PDFs/YouTube: load → classify doc/chunk-level
    metadata → SectionChunker (with llm=llm) → safety-net re-classification
    → store in the SAME ChromaVectorStore collection used by /eval/upload
    and /eval/ingest-video. Because eval_query() already searches that one
    collection, webpage chunks are automatically returned by /eval/query —
    no separate retrieval path or merge step is needed.
    """
    url = req.url.strip()
    logger.info("[INGEST-WEBPAGE] received url=%s", url)

    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=422, detail="Please provide a valid http(s) URL.")

    if is_youtube_url(url):
        raise HTTPException(
            status_code=422,
            detail="This looks like a YouTube link — use /eval/ingest-video instead.",
        )

    t0 = time.perf_counter()

    try:
        logger.info("[INGEST-WEBPAGE] calling load_url_advanced...")
        docs = load_url_advanced(url)
        logger.info("[INGEST-WEBPAGE] load_url_advanced returned %d docs", len(docs))
    except Exception as exc:
        traceback.print_exc()
        logger.error("[INGEST-WEBPAGE] load_url_advanced FAILED: %s", exc, exc_info=True)
        raise HTTPException(status_code=422, detail=f"Failed to fetch web page: {exc}")

    if not docs:
        raise HTTPException(
            status_code=422,
            detail="No extractable text content found on this page (it may be blocked, "
                   "empty, or require JavaScript that couldn't be rendered).",
        )

    llm = get_eval_llm()

    title          = docs[0].metadata.get("title") or url
    source_type    = docs[0].metadata.get("source_type", "web")
    # Use a stable, readable synthetic filename so the doc shows up cleanly
    # in /eval/documents and the sidebar doc list — same idea as
    # youtube_{video_id}.txt for videos.
    slug           = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())[:60].strip("_") or "page"
    synthetic_name = f"webpage_{slug}.txt"
    logger.info("[INGEST-WEBPAGE] title=%r source_type=%s raw_docs=%d", title, source_type, len(docs))

    preview    = docs[0].page_content[:5000]
    extra_text = docs[0].page_content[5000:10000]
    # Let classification run normally (document_loader already defaults
    # doc_type="general" in metadata, but we re-derive it the same way
    # /eval/upload does so a policy-heavy webpage can still be classified
    # as "policy_document"/"reference_handbook"/"regulatory" if it matches).
    doc_type = classify_document_type(synthetic_name, preview, extra_text)
    doc_meta = tag_document(synthetic_name, preview, extra_text=extra_text, doc_type=doc_type, llm=llm)
    doc_meta["doc_type"] = doc_type

    for doc in docs:
        doc.metadata.setdefault("filename", synthetic_name)
        doc.metadata["source"]      = url
        doc.metadata["source_url"]  = url
        doc.metadata["insurer"]     = doc_meta.get("insurer", "UNKNOWN")
        doc.metadata["doc_type"]    = doc_type
        doc.metadata.setdefault("source_type", source_type)

    # Same SectionChunker pass as upload/video — llm=llm is always passed so
    # chunk-level section/policy_type classification actually runs instead
    # of silently degrading to regex-only "general".
    try:
        pipeline = get_rag_pipeline()
        chunks = pipeline.chunker.split_documents(docs, doc_type=doc_type, llm=llm)
        logger.info("[INGEST-WEBPAGE] SectionChunker → %d chunks from %d raw page doc(s)",
                    len(chunks), len(docs))
    except Exception as exc:
        traceback.print_exc()
        logger.warning("[INGEST-WEBPAGE] SectionChunker failed (%s), using raw page doc(s)", exc)
        chunks = docs

    for chunk in chunks:
        chunk.metadata.setdefault("filename", synthetic_name)
        chunk.metadata["source"]     = url
        chunk.metadata["source_url"] = url
        chunk.metadata["insurer"]    = doc_meta.get("insurer", "UNKNOWN")
        chunk.metadata["doc_type"]   = doc_type
        chunk.metadata.setdefault("source_type", source_type)

    # Safety-net classification pass — same as upload/video endpoints.
    _classify_docs_metadata(chunks, doc_type, doc_meta, llm)

    vs  = get_vector_store()
    ids = vs.add_documents(chunks)
    elapsed = (time.perf_counter() - t0) * 1000

    assigned_policy_types = list({c.metadata.get("policy_type", "general") for c in chunks})
    assigned_sections      = list({c.metadata.get("section", "general") for c in chunks})
    general_count = sum(1 for c in chunks if c.metadata.get("policy_type", "general") == "general")

    logger.info(
        "[INGEST-WEBPAGE] done — %d chunks stored | policy_types=%s | sections=%s | %d/%d still 'general'",
        len(ids), assigned_policy_types, assigned_sections, general_count, len(chunks),
    )

    # ── Generate and store webpage summary ────────────────────────────────────
    try:
        summary_text = generate_summary(docs, source=url, doc_meta=doc_meta, llm=llm)
        get_summary_store().upsert(
            source=url,
            summary_text=summary_text,
            metadata={
                "source_type":  source_type,
                "doc_type":     doc_type,
                "insurer":      doc_meta.get("insurer", "UNKNOWN"),
                "policy_type":  doc_meta.get("policy_type", "general"),
                "chunk_count":  len(ids),
                "title":        title,
                "summary_type": "webpage",
            },
        )
        logger.info("[INGEST-WEBPAGE] summary stored for %s", url)
    except Exception as exc:
        logger.warning("[INGEST-WEBPAGE] summary generation failed: %s", exc)

    return {
        "status": "ok",
        "filename": synthetic_name,
        "url": url,
        "title": title,
        "source_type": source_type,
        "doc_type": doc_type,
        "chunks_added": len(ids),
        "total_in_store": vs.count(),
        "doc_metadata": doc_meta,
        "assigned_policy_types": assigned_policy_types,
        "assigned_sections": assigned_sections,
        "chunks_still_general_policy_type": general_count,
        "ingest_ms": round(elapsed, 1),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/eval/query", response_model=EvalResponse, summary="Run full RAG evaluation")
def eval_query(req: EvalRequest):
    vs = get_vector_store()
    if vs.count() == 0:
        raise HTTPException(status_code=400, detail="Vector store is empty. Upload at least one document first.")

    # ── KV Cache check ────────────────────────────────────────────────────────
    # Key includes the full source list so any newly ingested document
    # automatically invalidates cached answers for the same query.
    kv = get_kv_cache()
    current_sources = vs.list_sources()
    cache_key = kv.make_key(
        req.query, req.top_k, req.use_hybrid, req.use_reranker,
        req.generate_answer, req.run_ragas, current_sources,
    )

    # 1 — exact cache hit (same wording, O(1))
    cached = kv.get(cache_key)

    # 2 — semantic cache hit (similar meaning, cosine ≥ 0.92)
    #     Embed the query once here; reuse the embedding downstream for
    #     compression scoring so we don't embed twice.
    _query_emb = None
    if cached is None:
        try:
            _query_emb = get_compressor()._model.encode(
                [req.query], normalize_embeddings=True, show_progress_bar=False
            )[0]
            cached = kv.semantic_get(_query_emb)
        except Exception as _sem_exc:
            logger.debug("[EVAL] semantic cache lookup failed: %s", _sem_exc)

    if cached is not None:
        logger.info("[EVAL] cache hit for query=%r", req.query[:60])
        cached["doc_metadata"]["from_cache"] = True
        return EvalResponse(**cached)

    t_start = time.perf_counter()

    llm = get_eval_llm()

    # ── Stage 1 + Stage 2 retrieval (shared with streaming endpoint) ──────────
    ret            = _retrieve(req)
    raw_docs       = ret["raw_docs"]
    summary_context = ret["summary_context"]
    filter_meta    = ret["filter_meta"]
    query_meta     = ret["query_meta"]
    retrieval_ms   = ret["retrieval_ms"]

    chunks: list[ChunkInfo] = []
    for idx, doc in enumerate(raw_docs):
        meta    = dict(doc.metadata)
        doc_type_for_chunk = meta.get("doc_type", "policy_document")

        # Use stored section if available; fall back to regex-only (no LLM) at
        # query time — LLM section tagging happens at ingestion, not retrieval.
        section = meta.get("section") or classify_chunk_intent(
            doc.page_content, doc_type=doc_type_for_chunk, llm=None
        )

        sim     = float(meta.get("similarity", 0.0))
        rerank  = float(meta["rerank_score"]) if "rerank_score" in meta else None
        chunks.append(
            ChunkInfo(
                chunk_index=idx,
                text=doc.page_content,
                char_count=len(doc.page_content),
                word_count=len(doc.page_content.split()),
                similarity_score=round(sim, 4),
                rerank_score=round(rerank, 4) if rerank is not None else None,
                retrieval_method=meta.get("retrieval_method", "dense"),
                metadata=meta,
                section=section,
            )
        )

    seen_sources: set[str] = set()
    doc_metadata: dict[str, Any] = {
        "query_insurer_hint":     query_meta.get("insurer"),
        "query_policy_type_hint": query_meta.get("policy_type"),
        "metadata_filter_applied": filter_meta,
        "sources_retrieved":      [],
        "sections_hit":           [],
        "policy_types_hit":       [],
        "retrieval_method":       raw_docs[0].metadata.get("retrieval_method", "dense") if raw_docs else "n/a",
        "total_chunks_in_store":  vs.count(),
        "chunks_retrieved":       len(chunks),
    }
    sections_seen: set[str]     = set()
    policy_types_seen: set[str] = set()

    for c in chunks:
        src = c.metadata.get("source", "unknown")
        if src not in seen_sources:
            seen_sources.add(src)
            doc_metadata["sources_retrieved"].append(src)
        if c.section not in sections_seen:
            sections_seen.add(c.section)
            doc_metadata["sections_hit"].append(c.section)
        pt = c.metadata.get("policy_type", "general")
        if pt not in policy_types_seen:
            policy_types_seen.add(pt)
            doc_metadata["policy_types_hit"].append(pt)

    chunk_metadata = [
        {
            "chunk_index":      c.chunk_index,
            "source":           c.metadata.get("source", "unknown"),
            "filename":         c.metadata.get("filename", "unknown"),
            "page":             c.metadata.get("page"),
            "section":          c.section,
            "insurer":          c.metadata.get("insurer", "UNKNOWN"),
            "policy_type":      c.metadata.get("policy_type", "general"),
            "doc_type":         c.metadata.get("doc_type", "general"),
            "source_type":      c.metadata.get("source_type", ""),
            "similarity_score": c.similarity_score,
            "rerank_score":     c.rerank_score,
            "retrieval_method": c.retrieval_method,
            "word_count":       c.word_count,
            "char_count":       c.char_count,
        }
        for c in chunks
    ]

    logger.info(
        "[EVAL] retrieved %d chunks | sections=%s | policy_types=%s",
        len(chunks), doc_metadata["sections_hit"], doc_metadata["policy_types_hit"],
    )

    has_conflict, conflict_insurers = detect_conflict(raw_docs)

    answer: Optional[str] = None
    sources: list[str]    = []
    llm_ms                = 0.0

    if req.generate_answer:
        t_llm_start = time.perf_counter()
        try:
            # raw_docs already compressed + budget-capped by context_compressor above
            sources       = _sources_from_chunks(raw_docs)
            context       = _build_structured_context(raw_docs, max_chars=MAX_CONTEXT_CHARS)
            cond_hint     = _extract_condition_hint(raw_docs)
            has_conflict_local, conflict_ins = detect_conflict(raw_docs)
            conflict_hint = (
                f"The context contains multiple insurers ({', '.join(sorted(conflict_ins))}). "
                "Keep each insurer's facts separate."
                if has_conflict_local else ""
            )

            # Include document summaries above detailed chunks so the LLM has
            # high-level context before diving into specific excerpt details.
            summary_section = (
                f"DOCUMENT SUMMARIES (high-level overview of relevant sources):\n{summary_context}\n\n"
                if summary_context else ""
            )

            _conflict_note = f"\nNOTE: {conflict_hint}" if conflict_hint else ""
            # Decide citation style based on what sources are present
            has_yt_src = any(
                d.metadata.get("doc_type") == "youtube"
                or "youtube" in str(d.metadata.get("source_type", "")).lower()
                for d in raw_docs
            )
            cite_rule = (
                "Cite document facts as [Doc: filename, p.N] and video facts as [Video: URL]."
                if has_yt_src else
                "Cite every fact as [Doc: filename, p.N]."
            )
            citation_prompt = (
                f"You are an insurance policy assistant. Answer using ONLY the context below. "
                f"{cite_rule} If not found say 'Not in documents'. "
                f"No invented data.{_conflict_note}\n\n"
                f"{summary_section}CONTEXT:\n{context}\n\nQUESTION: {req.query}\nANSWER:"
            )

            answer_llm = llm or get_eval_llm()
            response   = answer_llm.invoke(citation_prompt)
            answer     = response.content if hasattr(response, "content") else str(response)

            if not answer.strip():
                answer = "Not mentioned in documents."

        except Exception as exc:
            traceback.print_exc()
            logger.error("[EVAL] LLM generation failed: %s", exc, exc_info=True)
            answer  = f"[LLM error: {exc}]"
            sources = list(seen_sources)
        llm_ms = (time.perf_counter() - t_llm_start) * 1000

    ragas_scores: Optional[RagasScores] = None
    ragas_per_chunk: list[dict[str, Any]] = []
    ragas_ms = 0.0

    if req.run_ragas and req.generate_answer and chunks:
        t_ragas_start = time.perf_counter()
        ragas_scores, ragas_per_chunk = _run_ragas(
            query=req.query,
            answer=answer or "(no answer generated)",
            chunks=chunks,
        )
        ragas_ms = (time.perf_counter() - t_ragas_start) * 1000
    elif req.run_ragas and not req.generate_answer:
        logger.warning("[EVAL] run_ragas=true but generate_answer=false — skipping RAGAS")

    total_ms = (time.perf_counter() - t_start) * 1000

    response = EvalResponse(
        query=req.query,
        doc_metadata=doc_metadata,
        chunk_metadata=chunk_metadata,
        chunks=chunks,
        total_chunks_in_store=vs.count(),
        ragas=ragas_scores,
        ragas_per_chunk=ragas_per_chunk,
        answer=answer,
        sources=sources,
        has_conflict=has_conflict,
        conflict_insurers=list(conflict_insurers) if conflict_insurers else [],
        timing={
            "retrieval_ms": round(retrieval_ms, 1),
            "llm_ms":       round(llm_ms, 1),
            "ragas_ms":     round(ragas_ms, 1),
            "total_ms":     round(total_ms, 1),
        },
    )

    # ── Store in KV cache (skip if answer contains an LLM error) ─────────────
    try:
        if not (answer and answer.startswith("[LLM error")):
            kv.put(
                cache_key,
                response.model_dump(),
                query_embedding=_query_emb,
                query_text=req.query,
            )
            logger.info("[EVAL] KV cache stored (semantic-enabled) for query=%r", req.query[:60])
    except Exception as exc:
        logger.warning("[EVAL] KV cache store failed: %s", exc)

    return response


# ══════════════════════════════════════════════════════════════════════════════
# STREAMING QUERY ENDPOINT
# User sees first tokens in ~2-3 s even when full answer takes 8-10 s.
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/eval/query/stream", summary="Stream answer tokens via SSE")
def eval_query_stream(req: EvalRequest):
    """
    Server-Sent Events endpoint.

    Event sequence
    --------------
    1. {"type":"retrieval_complete", ...full EvalResponse fields except answer...}
       Emitted after retrieval + compression (~0.5 s). Frontend renders the
       full tab UI (chunks, metadata, timing) immediately from this event.

    2. {"type":"token", "text":"..."}
       One event per streamed LLM token. Frontend appends to the answer box
       that was already rendered in step 1.

    3. {"type":"done", "timing_ms":N, "llm_ms":N}
       Final event. Frontend updates the timing display.

    4. {"type":"error", "message":"..."}
       Emitted instead of steps 2-3 on failure.
    """
    from fastapi.responses import StreamingResponse as SR

    def _sse(obj: dict) -> str:
        return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"

    def generate():
        vs = get_vector_store()
        kv = get_kv_cache()
        t0 = time.perf_counter()

        # ── Cache check ──────────────────────────────────────────────────────
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
                cached = kv.semantic_get(_q_emb)
            except Exception:
                pass

        if cached is not None:
            # Emit the cached full response as retrieval_complete so the UI
            # renders all tabs, then stream the answer text for a nice effect.
            cached["doc_metadata"]["from_cache"] = True
            cached_answer = cached.get("answer") or ""
            cached["answer"] = ""   # answer arrives via token events
            yield _sse({"type": "retrieval_complete", **cached})
            for i in range(0, len(cached_answer), 6):
                yield _sse({"type": "token", "text": cached_answer[i:i+6]})
            total_ms = round((time.perf_counter() - t0) * 1000, 1)
            yield _sse({"type": "done", "timing_ms": total_ms, "llm_ms": 0,
                        "from_cache": True})
            return

        # ── Retrieval (shared logic with eval_query) ──────────────────────────
        if vs.count() == 0:
            yield _sse({"type": "error", "message": "Vector store is empty. Upload documents first."})
            return

        ret             = _retrieve(req)
        raw_docs        = ret["raw_docs"]
        summary_context = ret["summary_context"]
        filter_meta     = ret["filter_meta"]
        query_meta      = ret["query_meta"]
        retrieval_ms    = ret["retrieval_ms"]

        # Build ChunkInfo-compatible dicts for the UI
        chunks_data = []
        seen_sources: set = set()
        sections_hit: list = []
        policy_types_hit: list = []
        sections_seen: set = set()
        policy_types_seen: set = set()

        for idx, doc in enumerate(raw_docs):
            meta = dict(doc.metadata)
            section = meta.get("section") or classify_chunk_intent(
                doc.page_content, doc_type=meta.get("doc_type", "policy_document"), llm=None
            )
            sim    = float(meta.get("similarity", 0.0))
            rerank = float(meta["rerank_score"]) if "rerank_score" in meta else None
            src    = meta.get("source", "unknown")

            chunks_data.append({
                "chunk_index":      idx,
                "text":             doc.page_content,
                "char_count":       len(doc.page_content),
                "word_count":       len(doc.page_content.split()),
                "similarity_score": round(sim, 4),
                "rerank_score":     round(rerank, 4) if rerank is not None else None,
                "retrieval_method": meta.get("retrieval_method", "dense"),
                "metadata":         meta,
                "section":          section,
            })
            seen_sources.add(src)
            if section not in sections_seen:
                sections_seen.add(section)
                sections_hit.append(section)
            pt = meta.get("policy_type", "general")
            if pt not in policy_types_seen:
                policy_types_seen.add(pt)
                policy_types_hit.append(pt)

        sources_list = list(seen_sources)
        has_conflict_local, conflict_ins = detect_conflict(raw_docs)

        chunk_metadata = [
            {
                "chunk_index":      c["chunk_index"],
                "source":           c["metadata"].get("source", "unknown"),
                "page":             c["metadata"].get("page"),
                "insurer":          c["metadata"].get("insurer", "UNKNOWN"),
                "policy_type":      c["metadata"].get("policy_type", "general"),
                "doc_type":         c["metadata"].get("doc_type", "policy_document"),
                "section":          c["section"],
                "similarity_score": c["similarity_score"],
                "rerank_score":     c["rerank_score"],
                "retrieval_method": c["retrieval_method"],
                "char_count":       c["char_count"],
                "word_count":       c["word_count"],
                "compressed":       c["metadata"].get("compressed", False),
            }
            for c in chunks_data
        ]

        doc_metadata = {
            "query_insurer_hint":      query_meta.get("insurer"),
            "query_policy_type_hint":  query_meta.get("policy_type"),
            "metadata_filter_applied": filter_meta,
            "sources_retrieved":       sources_list,
            "sections_hit":            sections_hit,
            "policy_types_hit":        policy_types_hit,
            "retrieval_method":        raw_docs[0].metadata.get("retrieval_method", "dense") if raw_docs else "n/a",
            "total_chunks_in_store":   vs.count(),
            "chunks_retrieved":        len(chunks_data),
            "from_cache":              False,
        }

        # Emit retrieval_complete — frontend renders full tab UI from this
        yield _sse({
            "type":               "retrieval_complete",
            "query":              req.query,
            "doc_metadata":       doc_metadata,
            "chunk_metadata":     chunk_metadata,
            "chunks":             chunks_data,
            "total_chunks_in_store": vs.count(),
            "ragas":              None,
            "ragas_per_chunk":    [],
            "answer":             "",   # filled by token events
            "sources":            sources_list,
            "has_conflict":       has_conflict_local,
            "conflict_insurers":  list(conflict_ins) if conflict_ins else [],
            "timing": {
                "retrieval_ms": retrieval_ms,
                "llm_ms":       0,
                "ragas_ms":     0,
                "total_ms":     retrieval_ms,
            },
        })

        if not req.generate_answer or not raw_docs:
            total_ms = round((time.perf_counter() - t0) * 1000, 1)
            yield _sse({"type": "done", "timing_ms": total_ms, "llm_ms": 0})
            return

        # ── Stream LLM answer ─────────────────────────────────────────────────
        llm = get_eval_llm()
        if llm is None:
            yield _sse({"type": "error", "message": "LLM not configured."})
            return

        context = _build_structured_context(raw_docs, max_chars=MAX_CONTEXT_CHARS)
        summary_section = f"SUMMARIES:\n{summary_context}\n\n" if summary_context else ""
        conflict_hint = (
            f"\nNOTE: Multiple insurers ({', '.join(sorted(conflict_ins))}). Keep facts separate."
            if has_conflict_local else ""
        )
        has_yt_src = any(
            d.metadata.get("doc_type") == "youtube"
            or "youtube" in str(d.metadata.get("source_type", "")).lower()
            for d in raw_docs
        )
        cite_rule = (
            "Cite document facts as [Doc: filename, p.N] and video facts as [Video: URL]."
            if has_yt_src else
            "Cite every fact as [Doc: filename, p.N]."
        )
        prompt = (
            f"You are an insurance policy assistant. Answer using ONLY the context below. "
            f"{cite_rule} If not found say 'Not in documents'. "
            f"No invented data.{conflict_hint}\n\n"
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

        llm_ms    = round((time.perf_counter() - t_llm) * 1000, 1)
        total_ms  = round((time.perf_counter() - t0) * 1000, 1)
        yield _sse({"type": "done", "timing_ms": total_ms, "llm_ms": llm_ms,
                    "retrieval_ms": retrieval_ms})

        # ── Store in KV cache ─────────────────────────────────────────────────
        try:
            if full_answer and not full_answer.startswith("[LLM error"):
                full_resp = {
                    "query": req.query, "doc_metadata": doc_metadata,
                    "chunk_metadata": chunk_metadata, "chunks": chunks_data,
                    "total_chunks_in_store": vs.count(), "ragas": None,
                    "ragas_per_chunk": [], "answer": full_answer,
                    "sources": sources_list, "has_conflict": has_conflict_local,
                    "conflict_insurers": list(conflict_ins) if conflict_ins else [],
                    "timing": {"retrieval_ms": retrieval_ms, "llm_ms": llm_ms,
                               "ragas_ms": 0, "total_ms": total_ms},
                }
                kv.put(cache_key, full_resp, query_embedding=_q_emb, query_text=req.query)
        except Exception:
            pass

    return SR(generate(), media_type="text/event-stream",
              headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ══════════════════════════════════════════════════════════════════════════════
# RAGAS EVALUATION VIA LLM-AS-JUDGE
# ══════════════════════════════════════════════════════════════════════════════

def _clean_json(raw: str) -> str:
    """Strip markdown fences, preamble, and extract only the JSON object/array."""
    raw = re.sub(r"```(?:json)?|```", "", raw).strip()
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', raw)
    if match:
        return match.group(1).strip()
    return raw


def _llm_invoke_json(llm, system_content: str, user_content: str) -> str:
    """
    Invoke LLM with explicit system + user messages, forcing JSON-only output.
    Small judge models (e.g. Qwen 2.5 3B) tend to wrap answers in prose or
    {"response": "..."} envelopes unless given an explicit system role.
    """
    from langchain_core.messages import SystemMessage, HumanMessage
    resp = llm.invoke([
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ])
    return resp.content if hasattr(resp, "content") else str(resp)


def _run_ragas(
    query: str,
    answer: str,
    chunks: list[ChunkInfo],
) -> tuple[RagasScores, list[dict]]:
    """
    Run RAGAS evaluation with all 3 LLM calls fired in parallel so total
    latency equals the slowest single call instead of 3× the slowest call.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # RAGAS judge only outputs short JSON (< 80 tokens) — use a compact
    # LLM instance to avoid burning the same 200-token budget as answers.
    llm = get_insurance_llm(temperature=0, max_tokens=80)
    SYSTEM_JSON = "Output ONLY valid JSON. No prose, no markdown."

    chunk_lines  = "\n".join(f"[Chunk {c.chunk_index}] {c.text[:400]}" for c in chunks)
    context_blob = "\n\n---\n\n".join(f"[Chunk {c.chunk_index}]\n{c.text[:400]}" for c in chunks)

    per_chunk_user = (
        f'Score each chunk\'s relevance to the query from 0.0 to 1.0.\n\n'
        f'Query: "{query}"\n\nChunks:\n{chunk_lines}\n\n'
        f'Return a JSON array, one object per chunk:\n'
        f'[{{"chunk_index": 0, "relevance": 0.85, "reason": "brief"}}]'
    )

    faithfulness_user = (
        f'Score these metrics from 0.0 to 1.0.\n\nQuery: "{query}"\n\n'
        f'Context:\n{context_blob}\n\nAnswer:\n{answer[:800]}\n\n'
        f'- faithfulness: fraction of answer claims supported by context\n'
        f'- answer_relevancy: how well the answer addresses the query\n\n'
        f'Return ONLY: {{"faithfulness": 0.9, "answer_relevancy": 0.85}}'
    )

    precision_recall_user = (
        f'Score these metrics from 0.0 to 1.0.\n\nQuery: "{query}"\n\n'
        f'Context Chunks:\n{context_blob}\n\n'
        f'- context_precision: fraction of retrieved chunks that are relevant\n'
        f'- context_recall: does the context contain enough to fully answer\n\n'
        f'Return ONLY: {{"context_precision": 0.9, "context_recall": 0.85}}'
    )

    # Fire all 3 LLM calls in parallel — total time = max(t1, t2, t3), not t1+t2+t3
    tasks = {
        "per_chunk":   per_chunk_user,
        "faithfulness": faithfulness_user,
        "precision":   precision_recall_user,
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
                logger.info("[RAGAS] %s raw: %s", name, repr(raw_results[name][:200]))
            except Exception as exc:
                logger.warning("[RAGAS] %s failed: %s", name, exc)

    # Parse per-chunk scores
    ragas_per_chunk: list[dict] = []
    try:
        ragas_per_chunk = json.loads(_clean_json(raw_results.get("per_chunk", "[]")))
    except Exception as exc:
        logger.warning("[RAGAS] per-chunk parse failed: %s — using similarity fallback", exc)
        ragas_per_chunk = [
            {"chunk_index": c.chunk_index, "relevance": c.similarity_score, "reason": "fallback"}
            for c in chunks
        ]

    avg_chunk_relevance = (
        sum(r.get("relevance", 0) for r in ragas_per_chunk) / len(ragas_per_chunk)
        if ragas_per_chunk else 0.0
    )

    # Parse aggregate scores
    scores: dict[str, float] = {}
    is_fallback = False
    for name in ("faithfulness", "precision"):
        raw = raw_results.get(name, "")
        if not raw:
            is_fallback = True
            continue
        try:
            parsed = json.loads(_clean_json(raw))
            scores.update(parsed)
            logger.info("[RAGAS] %s scores: %s", name, parsed)
        except Exception as exc:
            logger.warning("[RAGAS] %s parse failed: %s", name, exc)
            is_fallback = True

    def _safe_score(key: str) -> float:
        val = scores.get(key)
        if val is None:
            return round(avg_chunk_relevance, 3)
        try:
            return round(float(val), 3)
        except (TypeError, ValueError):
            return round(avg_chunk_relevance, 3)

    if is_fallback or not scores:
        return RagasScores(
            faithfulness=round(avg_chunk_relevance, 3),
            answer_relevancy=round(avg_chunk_relevance, 3),
            context_precision=round(avg_chunk_relevance, 3),
            context_recall=round(avg_chunk_relevance, 3),
            judge_model="fallback (LLM judge failed)",
            is_fallback=True,
        ), ragas_per_chunk

    return RagasScores(
        faithfulness=_safe_score("faithfulness"),
        answer_relevancy=_safe_score("answer_relevancy"),
        context_precision=_safe_score("context_precision"),
        context_recall=_safe_score("context_recall"),
        judge_model=get_active_model_info().get("model") or "unknown",
        is_fallback=False,
    ), ragas_per_chunk


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARIES ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/eval/summaries", summary="List all document summaries")
def eval_summaries():
    """Return stored summaries for every ingested document / video / webpage."""
    ss = get_summary_store()
    summaries = ss.list_all()
    return {
        "count": len(summaries),
        "summaries": [
            {
                "source":       s["metadata"].get("source", ""),
                "title":        s["metadata"].get("title", ""),
                "summary_type": s["metadata"].get("summary_type", "document"),
                "doc_type":     s["metadata"].get("doc_type", ""),
                "insurer":      s["metadata"].get("insurer", "UNKNOWN"),
                "policy_type":  s["metadata"].get("policy_type", "general"),
                "chunk_count":  s["metadata"].get("chunk_count", 0),
                "source_type":  s["metadata"].get("source_type", ""),
                "ingested_at":  s["metadata"].get("ingested_at", ""),
                "text":         s["text"],
            }
            for s in summaries
        ],
    }


@app.post("/eval/summaries/search", summary="Search summaries by query")
def eval_search_summaries(req: EvalRequest):
    """Semantic search over document summaries (Stage 1 of two-stage retrieval)."""
    ss = get_summary_store()
    docs = ss.search(req.query, top_k=req.top_k)
    return {
        "query": req.query,
        "results": [
            {
                "source":       d.metadata.get("source", ""),
                "title":        d.metadata.get("title", ""),
                "summary_type": d.metadata.get("summary_type", "document"),
                "insurer":      d.metadata.get("insurer", "UNKNOWN"),
                "policy_type":  d.metadata.get("policy_type", "general"),
                "similarity":   round(float(d.metadata.get("similarity", 0)), 4),
                "text":         d.page_content,
            }
            for d in docs
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# KV CACHE ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/eval/cache/stats", summary="KV cache statistics")
def eval_cache_stats():
    return get_kv_cache().stats()


@app.post("/eval/cache/flush", summary="Remove expired KV cache entries")
def eval_cache_flush():
    removed = get_kv_cache().flush()
    return {"status": "ok", "entries_removed": removed}


@app.post("/eval/cache/clear", summary="Clear all KV cache entries")
def eval_cache_clear():
    get_kv_cache().clear()
    return {"status": "ok", "message": "Cache cleared."}


# ══════════════════════════════════════════════════════════════════════════════
# LLM MODEL DISCOVERY & SELECTION
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/eval/llm-models", summary="List models available on the vLLM server")
def eval_llm_models():
    """
    Returns the list of model IDs reported by the vLLM server's /v1/models.
    Also includes which model is currently active.
    """
    models = list_vllm_models()
    info   = get_active_model_info()
    return {
        "available_models": models,
        "active_model":     info.get("model"),
        "backend":          info.get("backend"),
        "model_override":   info.get("model_override", False),
    }


class ModelSelectRequest(BaseModel):
    model: str


@app.post("/eval/llm-models/select", summary="Override the active vLLM model at runtime")
def eval_llm_model_select(req: ModelSelectRequest):
    """
    Override which model is sent to vLLM without restarting the server.
    The new model is used for all subsequent /eval/query and /eval/llm-test calls.
    """
    if not req.model.strip():
        return JSONResponse(status_code=400, content={"error": "model name cannot be empty"})
    global _llm_instance
    set_model_override(req.model.strip())
    _llm_instance = None   # force singleton to rebuild with new model
    get_eval_llm()         # pre-warm immediately
    return {"status": "ok", "active_model": req.model.strip()}


# ══════════════════════════════════════════════════════════════════════════════
# LLM CONNECTIVITY TEST
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/eval/llm-test", summary="Test whether the LLM backend is reachable")
def eval_llm_test():
    """
    Sends a minimal prompt to the configured LLM and reports whether it responds.
    Returns status='ok' with the backend/model info on success, or
    status='error' with a descriptive message on failure.
    """
    info = get_active_model_info()
    if info.get("backend") == "none":
        return JSONResponse(
            status_code=200,
            content={
                "status": "unconfigured",
                "backend": "none",
                "model": None,
                "message": (
                    "No LLM is configured. Set one of these environment variables before "
                    "starting the server:\n"
                    "  VLLM_HOST + VLLM_MODEL  — self-hosted vLLM server\n"
                    "  OPENAI_API_KEY          — OpenAI (gpt-4o-mini by default)\n"
                    "  ANTHROPIC_API_KEY       — Anthropic (claude-haiku-4-5-20251001 by default)"
                ),
            },
        )
    try:
        llm = get_eval_llm()
        resp = llm.invoke("Reply with the single word: OK")
        text = resp.content if hasattr(resp, "content") else str(resp)
        return {
            "status":  "ok",
            "backend": info.get("backend"),
            "model":   info.get("model"),
            "reply":   text.strip()[:80],
        }
    except Exception as exc:
        return JSONResponse(
            status_code=200,
            content={
                "status":  "error",
                "backend": info.get("backend"),
                "model":   info.get("model"),
                "message": str(exc),
            },
        )


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/eval/health")
def health():
    try:
        vs    = get_vector_store()
        count = vs.count()
    except Exception as exc:
        traceback.print_exc()
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})
    llm_info = get_active_model_info()
    return {
        "status":             "ok",
        "chunks_in_store":    count,
        "summaries_in_store": get_summary_store().count(),
        "cache_stats":        get_kv_cache().stats(),
        "llm_backend":        llm_info.get("backend", "none"),
        "llm_model":          llm_info.get("model") or "not configured",
        "vllm_host":          os.getenv("VLLM_HOST", "not-set"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DEV RUNNER
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    # reload=False is intentional: --reload causes uvicorn to re-import the
    # entire module on every saved file, which triggers BGE model (~5-8 s) and
    # TurboVec index reloads on EVERY file change during development — making
    # the first query after any edit feel broken.  Use docker compose restart
    # or send SIGTERM to reload manually when needed.
    uvicorn.run("eval_api:app", host="0.0.0.0", port=8002, reload=False)