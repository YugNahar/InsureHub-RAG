"""
RAG Evaluation API — eval_api.py
Drop this file into:  InsureHub-RAG-main/RAG_InsureAI/app/

Run with:
    cd InsureHub-RAG-main/RAG_InsureAI/app
    uvicorn eval_api:app --host 0.0.0.0 --port 8001 --reload

It connects directly to:
    • vector_store.py   → ChromaVectorStore (TurboVec-backed, hybrid BM25 + dense + reranker)
    • metadata_tagger.py → tag_document(), classify_query()
    • document_loader.py → load_document()
    • rag.py             → RAGPipeline (full answer generation with citations)
    • turbovec_store.py  → raw chunk scores (similarity + retrieval_method)
    • validator.py       → detect_conflict(), validate_grounding()

Evaluation metrics returned:
    1. metadata          — doc-level + chunk-level metadata for every retrieved chunk
    2. chunks            — raw chunk text + scores BEFORE LLM (no generation)
    3. ragas             — Faithfulness, Answer Relevancy, Context Precision, Context Recall
                          (computed via the configured LLM as judge)
    4. timing            — retrieval_ms, llm_ms, ragas_ms, total_ms
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Make sure the app/ directory is on the path ──────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.document_loader import load_document
from app.metadata_tagger import tag_document, classify_query
from app.vector_store import ChromaVectorStore
from app.rag import RAGPipeline, _detect_section, RETRIEVE_K, RERANK_K
from app.validator import detect_conflict, validate_grounding
from app.router import get_insurance_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="InsureHub RAG Evaluator",
    description="Evaluation dashboard API: metadata · raw chunks · RAGAS · timing",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared singletons (initialised once on first request) ─────────────────────
_vector_store: Optional[ChromaVectorStore] = None
_rag_pipeline: Optional[RAGPipeline] = None


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


# ── Request / Response models ──────────────────────────────────────────────────
class EvalRequest(BaseModel):
    query: str
    top_k: int = 5
    use_hybrid: bool = True
    use_reranker: bool = True
    generate_answer: bool = True   # set False to skip LLM and get chunks only
    run_ragas: bool = True         # set False to skip RAGAS scoring


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


class EvalResponse(BaseModel):
    query: str
    # 1. Metadata
    doc_metadata: dict[str, Any]
    chunk_metadata: list[dict[str, Any]]
    # 2. Raw chunks
    chunks: list[ChunkInfo]
    total_chunks_in_store: int
    # 3. RAGAS
    ragas: Optional[RagasScores]
    ragas_per_chunk: list[dict[str, Any]]
    # 4. Answer + timing
    answer: Optional[str]
    sources: list[str]
    has_conflict: bool
    conflict_insurers: list[str]
    timing: dict[str, float]


# ── Upload endpoint (delegates to existing RAG ingest) ────────────────────────
@app.post("/eval/upload", summary="Upload & ingest a document into the vector store")
async def eval_upload(file: UploadFile = File(...)):
    """
    Saves the uploaded file to a temp path, then calls load_document()
    (same as the main api.py does) and adds chunks to the shared ChromaVectorStore.
    """
    suffix = os.path.splitext(file.filename or "doc.pdf")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    t0 = time.perf_counter()
    try:
        docs = load_document(tmp_path, original_filename=file.filename or "document")
    except Exception as exc:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail=f"Document loading failed: {exc}")

    if not docs:
        os.unlink(tmp_path)
        raise HTTPException(status_code=422, detail="No extractable content found in document.")

    # Tag document-level metadata
    preview = " ".join(d.page_content for d in docs[:3])[:1000]
    doc_meta = tag_document(file.filename or "document", preview)

    # Inject metadata into each chunk
    for doc in docs:
        doc.metadata.setdefault("filename", file.filename)
        doc.metadata.setdefault("insurer", doc_meta.get("insurer", "UNKNOWN"))
        doc.metadata.setdefault("policy_type", doc_meta.get("policy_type", "general"))
        doc.metadata["section"] = _detect_section(doc.page_content)

    vs = get_vector_store()
    ids = vs.add_documents(docs)
    elapsed = (time.perf_counter() - t0) * 1000

    os.unlink(tmp_path)

    return {
        "status": "ok",
        "filename": file.filename,
        "chunks_added": len(ids),
        "total_in_store": vs.count(),
        "doc_metadata": doc_meta,
        "ingest_ms": round(elapsed, 1),
    }


# ── List documents in store ────────────────────────────────────────────────────
@app.get("/eval/documents", summary="List all ingested documents")
def eval_documents():
    vs = get_vector_store()
    return {
        "sources": vs.list_sources(),
        "filenames": vs.list_filenames(),
        "total_chunks": vs.count(),
    }


# ── Delete a document ──────────────────────────────────────────────────────────
@app.delete("/eval/documents/{source}", summary="Remove a document from the store")
def eval_delete(source: str):
    vs = get_vector_store()
    vs.delete_by_source(source)
    return {"status": "deleted", "source": source, "remaining_chunks": vs.count()}


# ── Main evaluation endpoint ───────────────────────────────────────────────────
@app.post("/eval/query", response_model=EvalResponse, summary="Run full RAG evaluation")
def eval_query(req: EvalRequest):
    """
    Given a query, returns:
      1. metadata   — document + chunk-level metadata for every retrieved chunk
      2. chunks     — raw text + scores before LLM
      3. ragas      — LLM-judged faithfulness / relevancy / precision / recall
      4. timing     — breakdown in milliseconds
    """
    vs = get_vector_store()
    if vs.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Vector store is empty. Upload at least one document first.",
        )

    t_start = time.perf_counter()

    # ── Step 1: Classify query & retrieve raw chunks ──────────────────────────
    query_meta = classify_query(req.query)
    logger.info("[EVAL] query_meta=%s", query_meta)

    t_ret_start = time.perf_counter()
    raw_docs = vs.search(
        query=req.query,
        top_k=req.top_k,
        use_hybrid=req.use_hybrid,
        use_reranker=req.use_reranker,
    )
    retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

    # ── Build ChunkInfo objects ────────────────────────────────────────────────
    chunks: list[ChunkInfo] = []
    for idx, doc in enumerate(raw_docs):
        meta = dict(doc.metadata)
        section = meta.get("section") or _detect_section(doc.page_content)
        sim = float(meta.get("similarity", 0.0))
        rerank = float(meta["rerank_score"]) if "rerank_score" in meta else None
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

    # ── Document-level metadata aggregated from retrieved chunks ──────────────
    seen_sources: set[str] = set()
    doc_metadata: dict[str, Any] = {
        "query_insurer_hint": query_meta.get("insurer"),
        "query_policy_type_hint": query_meta.get("policy_type"),
        "sources_retrieved": [],
        "sections_hit": [],
        "retrieval_method": raw_docs[0].metadata.get("retrieval_method", "dense") if raw_docs else "n/a",
        "total_chunks_in_store": vs.count(),
        "chunks_retrieved": len(chunks),
    }
    sections_seen: set[str] = set()
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        if src not in seen_sources:
            seen_sources.add(src)
            doc_metadata["sources_retrieved"].append(src)
        if c.section not in sections_seen:
            sections_seen.add(c.section)
            doc_metadata["sections_hit"].append(c.section)

    # Chunk-level metadata list (one per chunk, clean)
    chunk_metadata = [
        {
            "chunk_index": c.chunk_index,
            "source": c.metadata.get("source", "unknown"),
            "filename": c.metadata.get("filename", "unknown"),
            "page": c.metadata.get("page"),
            "section": c.section,
            "insurer": c.metadata.get("insurer", "UNKNOWN"),
            "policy_type": c.metadata.get("policy_type", "general"),
            "similarity_score": c.similarity_score,
            "rerank_score": c.rerank_score,
            "retrieval_method": c.retrieval_method,
            "word_count": c.word_count,
            "char_count": c.char_count,
        }
        for c in chunks
    ]

    # ── Conflict detection (from validator.py) ────────────────────────────────
    has_conflict, conflict_insurers = detect_conflict(raw_docs)

    # ── Step 2: Generate full answer (optional) ────────────────────────────────
    answer: Optional[str] = None
    sources: list[str] = []
    llm_ms = 0.0

    if req.generate_answer:
        t_llm_start = time.perf_counter()
        try:
            pipeline = get_rag_pipeline()
            answer, _, sources = pipeline.knowledge_query(req.query)
        except Exception as exc:
            logger.error("[EVAL] LLM generation failed: %s", exc)
            answer = f"[LLM error: {exc}]"
            sources = list(seen_sources)
        llm_ms = (time.perf_counter() - t_llm_start) * 1000

    # ── Step 3: RAGAS scoring (optional) ──────────────────────────────────────
    ragas_scores: Optional[RagasScores] = None
    ragas_per_chunk: list[dict[str, Any]] = []
    ragas_ms = 0.0

    if req.run_ragas and chunks:
        t_ragas_start = time.perf_counter()
        context_blob = "\n\n---\n\n".join(
            f"[Chunk {c.chunk_index}]\n{c.text}" for c in chunks
        )
        ragas_scores, ragas_per_chunk = _run_ragas(
            query=req.query,
            answer=answer or "(no answer generated)",
            chunks=chunks,
            context_blob=context_blob,
        )
        ragas_ms = (time.perf_counter() - t_ragas_start) * 1000

    total_ms = (time.perf_counter() - t_start) * 1000

    return EvalResponse(
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
            "llm_ms": round(llm_ms, 1),
            "ragas_ms": round(ragas_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    )


# ── RAGAS evaluation via LLM-as-judge ─────────────────────────────────────────
def _run_ragas(
    query: str,
    answer: str,
    chunks: list[ChunkInfo],
    context_blob: str,
) -> tuple[RagasScores, list[dict]]:
    """
    Uses the configured LLM (router.py → get_insurance_llm) as the judge.
    Computes the four core RAGAS metrics:
      • Faithfulness      — claims in answer supported by context
      • Answer relevancy  — answer addresses the actual question
      • Context precision — retrieved chunks are relevant to the query
      • Context recall    — context covers everything needed to answer

    Also computes a per-chunk relevance score.
    """
    llm = get_insurance_llm(temperature=0)

    # ── Per-chunk relevance ────────────────────────────────────────────────────
    per_chunk_prompt = f"""You are a RAG evaluation judge. For each chunk below, score its relevance to the query on a scale 0.0 to 1.0.

Query: "{query}"

{chr(10).join(f'[Chunk {c.chunk_index}] {c.text[:400]}' for c in chunks)}

Return ONLY a JSON array with one object per chunk:
[{{"chunk_index": 0, "relevance": 0.85, "reason": "..."}}, ...]
No markdown, no preamble."""

    ragas_per_chunk: list[dict] = []
    try:
        resp = llm.invoke(per_chunk_prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        import json, re as _re
        clean = _re.sub(r"```json|```", "", raw).strip()
        ragas_per_chunk = json.loads(clean)
    except Exception as exc:
        logger.warning("[RAGAS] per-chunk scoring failed: %s", exc)
        ragas_per_chunk = [
            {"chunk_index": c.chunk_index, "relevance": c.similarity_score, "reason": "fallback to similarity score"}
            for c in chunks
        ]

    avg_chunk_relevance = (
        sum(r.get("relevance", 0) for r in ragas_per_chunk) / len(ragas_per_chunk)
        if ragas_per_chunk
        else 0.0
    )

    # ── Main RAGAS metrics ─────────────────────────────────────────────────────
    ragas_prompt = f"""You are a RAGAS evaluation judge. Score the following on a scale 0.0 to 1.0.

Query: "{query}"

Retrieved Context:
{context_blob[:3000]}

Generated Answer:
{answer[:1500]}

Score these four metrics:
1. faithfulness       — Are ALL claims in the answer directly supported by the context? (1.0 = fully grounded)
2. answer_relevancy   — Does the answer directly address the query? (1.0 = perfectly on-topic)
3. context_precision  — Are the retrieved chunks relevant to the query? (1.0 = all chunks are useful)
4. context_recall     — Does the context contain everything needed to fully answer the query? (1.0 = complete coverage)

Return ONLY valid JSON:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0, "context_recall": 0.0, "reasoning": "..."}}
No markdown, no preamble."""

    try:
        resp = llm.invoke(ragas_prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        import json, re as _re2
        clean = _re2.sub(r"```json|```", "", raw).strip()
        scores = json.loads(clean)
        ragas_model = RagasScores(
            faithfulness=round(float(scores.get("faithfulness", avg_chunk_relevance)), 3),
            answer_relevancy=round(float(scores.get("answer_relevancy", avg_chunk_relevance)), 3),
            context_precision=round(float(scores.get("context_precision", avg_chunk_relevance)), 3),
            context_recall=round(float(scores.get("context_recall", avg_chunk_relevance * 0.9)), 3),
            judge_model=os.getenv("VLLM_MODEL", "configured-llm"),
        )
    except Exception as exc:
        logger.warning("[RAGAS] main scoring failed: %s — using fallback", exc)
        ragas_model = RagasScores(
            faithfulness=round(avg_chunk_relevance, 3),
            answer_relevancy=round(avg_chunk_relevance * 0.95, 3),
            context_precision=round(avg_chunk_relevance, 3),
            context_recall=round(avg_chunk_relevance * 0.9, 3),
            judge_model="fallback",
        )

    return ragas_model, ragas_per_chunk


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/eval/health")
def health():
    try:
        vs = get_vector_store()
        count = vs.count()
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})
    return {
        "status": "ok",
        "chunks_in_store": count,
        "llm_model": os.getenv("VLLM_MODEL", "not-set"),
        "vllm_host": os.getenv("VLLM_HOST", "not-set"),
    }


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("eval_api:app", host="0.0.0.0", port=8001, reload=True)