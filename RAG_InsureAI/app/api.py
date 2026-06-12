"""
FastAPI REST API — bridges the React/Lovable UI to the RAG pipeline.
Endpoints:
  POST   /upload            — async ingest document (returns job_id immediately)
  GET    /upload/{job_id}   — poll job status
  POST   /ask               — UNIFIED: answer from documents + videos + webpages (with memory AND stateful conversation)
  POST   /ask-documents-only— original (documents only) – for backward compatibility
  POST   /ask-stream        — streaming (documents only – kept unchanged)
  POST   /ask-url           — streaming URL question (fast, standalone)
  POST   /transcribe        — transcribe audio via Whisper
  GET    /docs              — list knowledge base documents
  DELETE /docs/{name}       — remove a specific document
  DELETE /docs              — clear all documents
  GET    /health            — health check
  POST   /upload-video      — store any video transcript permanently
  POST   /upload-webpage    — store webpage content permanently
  GET    /videos            — list stored video URLs
  DELETE /videos/{url}      — remove video
  GET    /webpages          — list stored webpage URLs
  DELETE /webpages/{url}    — remove webpage
  DELETE /conversation/{session_id} — clear conversation history
  POST   /conversation/reset/{session_id} — reset conversational state
"""
import asyncio
import logging
import os
import sys
import tempfile
import time
import uuid
import aiohttp
import re
from bs4 import BeautifulSoup
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import APIConnectionError, APIStatusError, APITimeoutError
from pydantic import BaseModel
import json as _json

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_JOB_TTL = 3600  # seconds — jobs older than this are pruned from memory

# Conversation memory: session_id -> list of {"role": "user/assistant", "content": "..."}
_conversations: dict[str, list[dict]] = {}
_MAX_HISTORY_TURNS = 3  # keep last 3 exchanges to stay within token limit


@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(_get_whisper())
    yield


app = FastAPI(title="InsureAI RAG API", docs_url="/swagger", redoc_url="/redoc", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Existing imports ──────────────────────────────────────────────────────────
from rag import RAGPipeline

_pipeline: RAGPipeline | None = None

_jobs: dict[str, dict] = {}
_ingest_semaphore: asyncio.Semaphore | None = None


def _get_ingest_semaphore() -> asyncio.Semaphore:
    global _ingest_semaphore
    if _ingest_semaphore is None:
        _ingest_semaphore = asyncio.Semaphore(1)
    return _ingest_semaphore


def _get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


def _job_state(status: str, filename: str, chunks: int = 0, error: str | None = None) -> dict:
    job = {"status": status, "filename": filename, "chunks": chunks, "_ts": time.time()}
    if error:
        job["error"] = error
    return job


def _prune_jobs() -> None:
    cutoff = time.time() - _JOB_TTL
    stale = [jid for jid, j in _jobs.items() if j.get("_ts", 0) < cutoff]
    for jid in stale:
        del _jobs[jid]
    if stale:
        logger.info("Pruned %d stale jobs from _jobs cache.", len(stale))


def _describe_llm_failure(exc: Exception) -> tuple[int, str]:
    from router import get_active_model_info

    model_info = get_active_model_info()
    backend = model_info["backend"]
    model = model_info["model"]
    if isinstance(exc, APITimeoutError):
        return 504, f"The AI model server at {backend} timed out while using {model}. Please try again in a moment."
    if isinstance(exc, APIConnectionError):
        return 502, f"The backend API is running, but it could not connect to the AI model server at {backend} while using {model}."
    if isinstance(exc, APIStatusError):
        status_code = getattr(exc, "status_code", "unknown")
        return 502, f"The AI model server at {backend} returned HTTP {status_code} while using {model}."
    return 500, "The backend could not generate an answer due to an unexpected internal error."


def _ingest_file(tmp_path: str, filename: str) -> int:
    from document_loader import load_document
    from metadata_tagger import tag_document

    pipeline = _get_pipeline()
    raw_docs = load_document(tmp_path, filename)
    chunks = pipeline._chunker.split_documents(raw_docs)
    pipeline._vector_store.delete_by_source(filename)
    preview = raw_docs[0].page_content[:600] if raw_docs else ""
    doc_tags = tag_document(filename, preview)
    for chunk in chunks:
        chunk.metadata["source"] = filename
        chunk.metadata.update(doc_tags)
    pipeline._vector_store.add_documents(chunks)
    return len(chunks)


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"  # optional — frontend can omit it


class URLRequest(BaseModel):
    url: str


# ══════════════════════════════════════════════════════════════════════════════
# MultiSourceRAG, VideoStore, WebpageStore
# ══════════════════════════════════════════════════════════════════════════════
from multi_source_rag import MultiSourceRAG
from document_loader import load_url_advanced, is_youtube_url, _load_youtube
from rag import SectionChunker

_multi_rag: MultiSourceRAG | None = None


def _get_multi_rag() -> MultiSourceRAG:
    global _multi_rag
    if _multi_rag is None:
        _multi_rag = MultiSourceRAG()
    return _multi_rag


def _chunk_transcript(transcript_text: str, url: str, title: str = "") -> list:
    from langchain_core.documents import Document

    chunker = SectionChunker(chunk_size=600, chunk_overlap=80)
    doc = Document(
        page_content=transcript_text,
        metadata={"source_url": url, "title": title, "type": "video_transcript"},
    )
    chunks = chunker.split_documents([doc])
    for chunk in chunks:
        chunk.metadata["source_type"] = "video"
        chunk.metadata["source_url"] = url
    return chunks


# ── Upload Video (any video URL) ───────────────────────────────────────────────────
@app.post("/upload-video")
async def upload_video(req: URLRequest):
    url = req.url.strip()
    multi = _get_multi_rag()
    if multi.video_exists(url):
        return {"status": "already_exists", "url": url, "message": "Video already in knowledge base."}
    try:
        from document_loader import load_url

        docs = await asyncio.to_thread(load_url, url)
        if not docs or not docs[0].page_content.strip():
            raise HTTPException(status_code=400, detail="Could not extract transcript from this video.")
        transcript_text = docs[0].page_content
        title = docs[0].metadata.get("title", url)
        chunks = _chunk_transcript(transcript_text, url, title)
        multi.add_video_chunks(url, chunks)
        return {"status": "success", "url": url, "chunks": len(chunks)}
    except Exception as exc:
        logger.exception("Video upload failed")
        raise HTTPException(status_code=500, detail=f"Video ingestion failed: {exc}")


# ── Upload Webpage (permanent) ───────────────────────────────────────────────
@app.post("/upload-webpage")
async def upload_webpage(req: URLRequest):
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL.")
    multi = _get_multi_rag()
    if multi.webpage_exists(url):
        return {"status": "already_exists", "url": url, "message": "Webpage already in knowledge base."}
    try:
        docs = await asyncio.to_thread(load_url_advanced, url)
        if not docs or len(docs[0].page_content.strip()) < 200:
            raise HTTPException(status_code=400, detail="Could not extract meaningful content from this URL.")
        chunker = SectionChunker(chunk_size=600, chunk_overlap=80)
        chunks = chunker.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source_type"] = "webpage"
            chunk.metadata["source_url"] = url
        multi.add_webpage_chunks(url, chunks)
        return {"status": "success", "url": url, "chunks": len(chunks)}
    except Exception as exc:
        logger.exception("Webpage upload failed")
        raise HTTPException(status_code=500, detail=f"Webpage ingestion failed: {exc}")


# ── List videos ──────────────────────────────────────────────────────────────
@app.get("/videos")
async def list_videos():
    multi = _get_multi_rag()
    return {"videos": multi.list_videos()}


# ── Delete video ─────────────────────────────────────────────────────────────
@app.delete("/videos/{url:path}")
async def delete_video(url: str):
    multi = _get_multi_rag()
    if not multi.video_exists(url):
        raise HTTPException(status_code=404, detail="Video URL not found.")
    multi.delete_video(url)
    return {"removed": True, "url": url}


# ── List webpages ────────────────────────────────────────────────────────────
@app.get("/webpages")
async def list_webpages():
    multi = _get_multi_rag()
    return {"webpages": multi.list_webpages()}


# ── Delete webpage ───────────────────────────────────────────────────────────
@app.delete("/webpages/{url:path}")
async def delete_webpage(url: str):
    multi = _get_multi_rag()
    if not multi.webpage_exists(url):
        raise HTTPException(status_code=404, detail="Webpage URL not found.")
    multi.delete_webpage(url)
    return {"removed": True, "url": url}


# ══════════════════════════════════════════════════════════════════════════════
# Conversation Agent (new)
# ══════════════════════════════════════════════════════════════════════════════
from conversation_agent import ConversationAgent

_conversation_agent: ConversationAgent | None = None


def _get_conversation_agent() -> ConversationAgent:
    global _conversation_agent
    if _conversation_agent is None:
        _conversation_agent = ConversationAgent(_get_pipeline()._vector_store, _get_multi_rag())
    return _conversation_agent


# ══════════════════════════════════════════════════════════════════════════════
# UNIFIED ASK (documents + videos + webpages) – WITH CONVERSATION MEMORY AND STATE
# ══════════════════════════════════════════════════════════════════════════════
@app.post("/ask")
async def ask(req: AskRequest):
    """
    UNIFIED answer from all sources: documents, videos, and webpages.
    Supports conversation memory via session_id AND stateful multi-turn recommendations.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Retrieve or create conversation history for this session
    history_list = _conversations.get(req.session_id, [])
    history_str = ""
    for turn in history_list[-_MAX_HISTORY_TURNS * 2 :]:
        history_str += f"{turn['role'].capitalize()}: {turn['content']}\n"

    agent = _get_conversation_agent()
    try:
        result, is_complete = await agent.process_message(req.session_id, req.question, history_str)

        answer = result.get("message", "")
        options = result.get("options", [])  # list of {id, label, description, recommended}

        # Update conversation memory
        history_list.append({"role": "user", "content": req.question})
        history_list.append({"role": "assistant", "content": answer})
        _conversations[req.session_id] = history_list[-_MAX_HISTORY_TURNS * 2 :]

        return {
            "answer": answer,
            "options": options,
            "sources": [],
            "conversation_continues": not is_complete,
        }
    except Exception as exc:
        logger.exception("Conversational ask failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/conversation/reset/{session_id}")
async def reset_conversation(session_id: str):
    """Reset the conversational agent's state for a session (clears pending questions)."""
    agent = _get_conversation_agent()
    agent.reset_session(session_id)
    # Also clear the stored conversation history if desired
    if session_id in _conversations:
        del _conversations[session_id]
    return {"status": "reset"}


@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history (legacy endpoint)."""
    if session_id in _conversations:
        del _conversations[session_id]
    # Also reset agent state
    _get_conversation_agent().reset_session(session_id)
    return {"status": "cleared"}


# ── Original document‑only ask (backward compatibility) ──────────────────────
@app.post("/ask-documents-only")
async def ask_documents_only(req: AskRequest):
    """
    Legacy endpoint: answers only from uploaded documents (no videos/webpages).
    Note: This also requires session_id but does not use history.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    try:
        answer, _, _ = await asyncio.wait_for(
            asyncio.to_thread(_get_pipeline().knowledge_query, req.question),
            timeout=150,
        )
        return {"answer": answer}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="The AI model server is taking too long to respond. Please try again in a moment.")
    except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
        logger.warning("Ask failed due to upstream model error: %s", exc)
        status_code, detail = _describe_llm_failure(exc)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except Exception as exc:
        logger.exception("Ask failed unexpectedly")
        raise HTTPException(status_code=500, detail=f"The backend failed while generating the answer: {exc}") from exc


# ══════════════════════════════════════════════════════════════════════════════
# ALL ORIGINAL ENDPOINTS REMAIN UNCHANGED BELOW
# ══════════════════════════════════════════════════════════════════════════════

# ── Upload (async) ────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    _prune_jobs()
    suffix = os.path.splitext(file.filename or "")[1].lower()
    supported = {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".csv", ".txt", ".eml"}
    if suffix not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    content = await file.read()
    filename = file.filename
    job_id = str(uuid.uuid4())
    _jobs[job_id] = _job_state("queued", filename)

    async def _process():
        tmp_path = None
        try:
            async with _get_ingest_semaphore():
                _jobs[job_id] = _job_state("processing", filename)
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                chunks = await asyncio.to_thread(_ingest_file, tmp_path, filename)
                _jobs[job_id] = _job_state("done", filename, chunks=chunks)
                logger.info("Ingested %s — %d chunks", filename, chunks)
        except Exception as exc:
            _jobs[job_id] = _job_state("error", filename, error=str(exc))
            logger.exception("Ingest failed for %s", filename)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    asyncio.create_task(_process())
    return {"job_id": job_id, "filename": filename, "status": "queued"}


@app.get("/upload/{job_id}")
async def upload_status(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


# ── Ingest URL (async) ────────────────────────────────────────────────────────
@app.post("/ingest-url")
async def ingest_url(req: URLRequest):
    _prune_jobs()
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL — must start with http:// or https://")
    job_id = str(uuid.uuid4())
    _jobs[job_id] = _job_state("queued", url)

    async def _process():
        try:
            _jobs[job_id] = _job_state("processing", url)
            chunks = await asyncio.to_thread(_get_pipeline().add_url, url)
            _jobs[job_id] = _job_state("done", url, chunks=chunks)
            logger.info("Ingested URL %s — %d chunks", url, chunks)
        except Exception as exc:
            _jobs[job_id] = _job_state("error", url, error=str(exc))
            logger.exception("URL ingest failed for %s", url)

    asyncio.create_task(_process())
    return {"job_id": job_id, "url": url, "status": "queued"}


# ── Ask with URL (ultra‑fast) ────────────────────────────────────────────────
class AskURLRequest(BaseModel):
    url: str
    question: str


async def fetch_url_text_async(url: str, max_chars: int = 1500) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"}) as resp:
                html = await resp.text()
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
        return text[:max_chars] if text else ""
    except Exception as e:
        logger.error(f"URL fetch error: {e}")
        return ""


@app.post("/ask-url")
async def ask_url(req: AskURLRequest):
    url = req.url.strip()
    question = req.question.strip() or "Summarize the content of this page."

    async def generate():
        context = await fetch_url_text_async(url, max_chars=1500)
        if not context:
            yield f"data: {_json.dumps({'error': 'Could not extract content from this URL.'})}\n\n".encode()
            return
        prompt = (
            "You are a helpful assistant. Answer based on the text below. Be concise (max 200 words).\n\n"
            f"Text: {context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        from router import VLLM_HOST, VLLM_MODEL
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model=VLLM_MODEL,
            base_url=f"{VLLM_HOST}/v1",
            api_key="EMPTY",
            temperature=0.3,
            max_tokens=200,
            timeout=25,
            max_retries=1,
        )
        try:
            response = await asyncio.to_thread(llm.invoke, [HumanMessage(content=prompt)])
            answer = response.content.strip()
            yield f"data: {_json.dumps({'answer': answer})}\n\n".encode()
        except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
            _, detail = _describe_llm_failure(exc)
            yield f"data: {_json.dumps({'error': detail})}\n\n".encode()
        except Exception as exc:
            logger.exception("URL answer generation failed")
            yield f"data: {_json.dumps({'error': f'Unexpected server error: {exc}'})}\n\n".encode()

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Streaming ask endpoint (documents only, unchanged) ────────────────────────
@app.post("/ask-stream")
async def ask_stream(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def generate():
        pipeline = _get_pipeline()
        try:
            answer, _, _ = await asyncio.to_thread(pipeline.knowledge_query, req.question)
            for chunk in [answer[i:i+30] for i in range(0, len(answer), 30)]:
                yield chunk
                await asyncio.sleep(0.01)
        except Exception as exc:
            yield f"Error: {exc}"

    return StreamingResponse(generate(), media_type="text/plain")


# ── Docs management ───────────────────────────────────────────────────────────
@app.get("/docs")
async def list_docs():
    pipeline = _get_pipeline()
    sources = pipeline.list_documents()
    counts = {}
    try:
        all_meta = pipeline._vector_store.collection.get(include=["metadatas"])
        for meta in all_meta["metadatas"]:
            src = meta.get("source")
            if src:
                counts[src] = counts.get(src, 0) + 1
    except Exception:
        pass
    documents = [f"{s} ({counts.get(s, 0)} chunks)" for s in sources]
    return {"documents": documents}


@app.delete("/docs")
async def clear_docs():
    await asyncio.to_thread(_get_pipeline().clear_documents)
    return {"status": "cleared"}


@app.delete("/docs/{name:path}")
async def remove_doc(name: str):
    pipeline = _get_pipeline()
    sources_before = set(pipeline.list_documents())
    await asyncio.to_thread(pipeline.remove_document, name)
    if name not in sources_before:
        raise HTTPException(status_code=404, detail=f"Document '{name}' not found.")
    return {"removed": True, "filename": name}


# ── Voice transcription (Whisper) – unchanged ────────────────────────────────
_whisper_lib = None
_whisper_model = None
_whisper_lock = asyncio.Lock()


def _import_whisper():
    global _whisper_lib
    if _whisper_lib is None:
        try:
            import whisper as _w

            _whisper_lib = _w
        except ImportError:
            raise RuntimeError("openai-whisper is not installed. Run: pip install openai-whisper")


def _load_whisper() -> None:
    global _whisper_model
    _import_whisper()
    logger.info("Loading Whisper model (base)...")
    _whisper_model = _whisper_lib.load_model("base")
    logger.info("Whisper model loaded.")


async def _get_whisper():
    global _whisper_model
    async with _whisper_lock:
        if _whisper_model is None:
            await asyncio.to_thread(_load_whisper)
    return _whisper_model


@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "recording.webm")[1].lower()
    if suffix not in {".webm", ".wav", ".mp3", ".m4a"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Use webm, wav, mp3, or m4a.")
    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    try:
        model = await _get_whisper()
        result = await asyncio.to_thread(model.transcribe, tmp_path)
        text = result["text"].strip()
        if not text:
            logger.warning("Transcription returned empty text.")
            return {"text": ""}
        logger.info("Transcribed: %s", text[:100])
        return {"text": text}
    except Exception as exc:
        logger.error("Whisper transcription failed: %s", exc)
        raise HTTPException(status_code=500, detail="Transcription failed.")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    pipeline = _get_pipeline()
    return {"status": "ok", "chunks": pipeline._vector_store.count()}