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
import threading
import time
import uuid
import aiohttp
import re
from bs4 import BeautifulSoup

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
_MAX_HISTORY_TURNS = 3  # keep last 3 exchanges to stay within token limit
_UPLOAD_READ_CHUNK_BYTES = 1024 * 1024


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; using %d.", name, raw, default)
        return default


_ASK_TIMEOUT_SECONDS = _int_env("ASK_TIMEOUT_SECONDS", 150)

_STATE_DIR = os.getenv("API_STATE_DIR", os.path.join(os.path.dirname(__file__), "state"))
_CONVERSATIONS_PATH = os.path.join(_STATE_DIR, "conversations.json")
_AGENT_SESSIONS_PATH = os.path.join(_STATE_DIR, "conversation_agent_sessions.json")
_JOBS_PATH = os.path.join(_STATE_DIR, "jobs.json")
_STATE_LOCK = threading.RLock()       # for sync background threads only
_ASYNC_STATE_LOCK: asyncio.Lock | None = None  # for async endpoint handlers


def _get_async_lock() -> asyncio.Lock:
    global _ASYNC_STATE_LOCK
    if _ASYNC_STATE_LOCK is None:
        _ASYNC_STATE_LOCK = asyncio.Lock()
    return _ASYNC_STATE_LOCK


_DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8501",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:8080",
]


def _csv_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "")
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or default


def _atomic_write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = f"{path}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as state_file:
            _json.dump(data, state_file, ensure_ascii=False)
            state_file.flush()
            os.fsync(state_file.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def _load_json_state(path: str, label: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as state_file:
            payload = _json.load(state_file)
        if not isinstance(payload, dict):
            raise ValueError("state root must be an object")
        return payload
    except FileNotFoundError:
        return {}
    except Exception as exc:
        logger.warning("Could not load %s state from %s: %s", label, path, exc)
        return {}


def _normalize_history(raw_turns: object) -> list[dict]:
    if not isinstance(raw_turns, list):
        return []
    history = []
    for turn in raw_turns:
        if not isinstance(turn, dict):
            continue
        role = str(turn.get("role", ""))
        content = str(turn.get("content", ""))
        if role in {"user", "assistant"}:
            history.append({"role": role, "content": content})
    return history[-_MAX_HISTORY_TURNS * 2 :]


def _load_conversations() -> dict[str, list[dict]]:
    payload = _load_json_state(_CONVERSATIONS_PATH, "conversation")
    return {
        str(session_id): history
        for session_id, turns in payload.items()
        if (history := _normalize_history(turns))
    }


def _load_agent_sessions() -> dict[str, dict]:
    payload = _load_json_state(_AGENT_SESSIONS_PATH, "conversation agent")
    return {
        str(session_id): dict(session)
        for session_id, session in payload.items()
        if isinstance(session, dict)
    }


def _normalize_job(raw_job: object) -> dict | None:
    if not isinstance(raw_job, dict):
        return None
    try:
        chunks = int(raw_job.get("chunks", 0) or 0)
    except (TypeError, ValueError):
        chunks = 0
    try:
        timestamp = float(raw_job.get("_ts", time.time()) or time.time())
    except (TypeError, ValueError):
        timestamp = time.time()

    status = str(raw_job.get("status", "error"))
    job = {
        "status": status,
        "filename": str(raw_job.get("filename", "")),
        "chunks": chunks,
        "_ts": timestamp,
    }
    if raw_job.get("error"):
        job["error"] = str(raw_job["error"])
    if status in {"queued", "processing"}:
        job["status"] = "error"
        job["error"] = "Server restarted before this background job finished. Please retry the upload."
        job["_ts"] = time.time()
    return job


def _load_jobs() -> dict[str, dict]:
    payload = _load_json_state(_JOBS_PATH, "job")
    jobs = {}
    changed = False
    for job_id, raw_job in payload.items():
        job = _normalize_job(raw_job)
        if job is None:
            changed = True
            continue
        if isinstance(raw_job, dict) and raw_job.get("status") in {"queued", "processing"}:
            changed = True
        jobs[str(job_id)] = job
    if changed:
        try:
            _atomic_write_json(_JOBS_PATH, jobs)
        except OSError as exc:
            logger.warning("Could not persist normalized job state: %s", exc)
    return jobs


# Conversation memory: session_id -> list of {"role": "user/assistant", "content": "..."}
_conversations: dict[str, list[dict]] = _load_conversations()
_agent_sessions: dict[str, dict] = _load_agent_sessions()
_jobs: dict[str, dict] = _load_jobs()


def _persist_conversations_locked() -> None:
    _atomic_write_json(_CONVERSATIONS_PATH, _conversations)


def _persist_agent_sessions_locked() -> None:
    _atomic_write_json(_AGENT_SESSIONS_PATH, _agent_sessions)


def _persist_jobs_locked() -> None:
    _atomic_write_json(_JOBS_PATH, _jobs)


async def _get_conversation_history(session_id: str) -> list[dict]:
    async with _get_async_lock():
        return [dict(turn) for turn in _conversations.get(session_id, [])]


async def _save_conversation_history(session_id: str, history: list[dict]) -> None:
    async with _get_async_lock():
        _conversations[session_id] = history[-_MAX_HISTORY_TURNS * 2 :]
        _persist_conversations_locked()


async def _delete_conversation_history(session_id: str) -> None:
    async with _get_async_lock():
        if session_id in _conversations:
            del _conversations[session_id]
            _persist_conversations_locked()


async def _save_agent_sessions(sessions: dict[str, dict]) -> None:
    async with _get_async_lock():
        _agent_sessions.clear()
        _agent_sessions.update({str(key): dict(value) for key, value in sessions.items()})
        _persist_agent_sessions_locked()


async def _delete_agent_session(session_id: str) -> None:
    async with _get_async_lock():
        if session_id in _agent_sessions:
            del _agent_sessions[session_id]
            _persist_agent_sessions_locked()


async def _set_job(job_id: str, job: dict) -> None:
    async with _get_async_lock():
        _jobs[job_id] = job
        _persist_jobs_locked()


async def _get_job(job_id: str) -> dict | None:
    async with _get_async_lock():
        job = _jobs.get(job_id)
        return dict(job) if job else None


app = FastAPI(title="InsureAI RAG API", docs_url="/swagger", redoc_url="/redoc")


@app.on_event("startup")
async def _init_async_lock():
    global _ASYNC_STATE_LOCK
    _ASYNC_STATE_LOCK = asyncio.Lock()


@app.on_event("startup")
async def _start_job_pruner():
    """Periodically prune stale jobs from memory (every 5 minutes)."""
    async def _prune_loop():
        while True:
            await asyncio.sleep(300)  # 5 minutes
            try:
                await _prune_jobs()
            except Exception:
                logger.exception("Periodic job pruner failed")
    app.state.job_pruner_task = asyncio.create_task(_prune_loop())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_csv_env("CORS_ALLOW_ORIGINS", _DEFAULT_CORS_ORIGINS),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Existing imports ──────────────────────────────────────────────────────────
from rag import RAGPipeline

_pipeline: RAGPipeline | None = None

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


async def _prune_jobs() -> None:
    cutoff = time.time() - _JOB_TTL
    async with _get_async_lock():
        stale = [jid for jid, j in _jobs.items() if j.get("_ts", 0) < cutoff]
        for jid in stale:
            del _jobs[jid]
        if stale:
            _persist_jobs_locked()
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
    from metadata_tagger import tag_document, classify_document_type

    pipeline = _get_pipeline()
    raw_docs = load_document(tmp_path, filename)

    llm = None
    try:
        llm = pipeline._get_llm()
    except Exception:
        pass

    # Use a unique source key per upload so identical filenames don't collide.
    upload_id = uuid.uuid4().hex[:12]
    unique_source = f"{upload_id}_{filename}"

    # ── Step 1: Classify document type BEFORE tagging ─────────────────────────
    # Use up to 5 000 chars of preview + next 3 pages so the classifier has
    # enough signal (legal handbooks often start with a table of contents, with
    # the substantive text beginning only on later pages).
    preview = raw_docs[0].page_content[:5000] if raw_docs else ""
    extra_text = " ".join(d.page_content for d in raw_docs[1:4])[:5000]
    doc_type = classify_document_type(filename, preview, extra_text)
    logger.info("Document type for '%s': %s", filename, doc_type)

    # ── Step 2: Tag document (skips keyword matching for non-policy docs) ──────
    doc_tags = tag_document(filename, preview, extra_text=extra_text, doc_type=doc_type, llm=llm)

    # ── Step 3: Annotate raw docs so the chunker inherits doc_type ────────────
    for raw_doc in raw_docs:
        raw_doc.metadata["doc_type"] = doc_type

    # ── Step 4: Chunk with doc-type-aware section detection ───────────────────
    chunks = pipeline.chunker.split_documents(raw_docs, doc_type=doc_type)

    # ── Step 5: Attach source, filename, and tags to every chunk ─────────────
    for chunk in chunks:
        chunk.metadata["source"] = unique_source
        chunk.metadata["filename"] = filename
        chunk.metadata.update(doc_tags)
        # Guarantee doc_type is not overwritten by doc_tags (it is set there
        # too, but be explicit for clarity and future-proofing).
        chunk.metadata["doc_type"] = doc_type

    pipeline.vector_store.add_documents(chunks)
    return len(chunks)


def _file_too_large_detail(filename: str, size: int) -> str:
    max_mb = MAX_FILE_SIZE_BYTES / (1024 * 1024)
    size_mb = size / (1024 * 1024)
    return f"File '{filename}' is too large ({size_mb:.1f} MB). Maximum allowed: {max_mb:.0f} MB."


async def _save_upload_to_temp(file: UploadFile, suffix: str, filename: str) -> str:
    tmp_path = None
    bytes_read = 0
    try:
        size_header = file.headers.get("content-length")
        if size_header:
            try:
                reported_size = int(size_header)
            except ValueError:
                reported_size = 0
            if reported_size > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=_file_too_large_detail(filename, reported_size),
                )
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            while True:
                chunk = await file.read(_UPLOAD_READ_CHUNK_BYTES)
                if not chunk:
                    break
                bytes_read += len(chunk)
                if bytes_read > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=_file_too_large_detail(filename, bytes_read),
                    )
                tmp.write(chunk)
        return tmp_path
    except Exception:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise
    finally:
        await file.close()


class AskRequest(BaseModel):
    question: str
    session_id: str = "default"  # optional — frontend can omit it


class URLRequest(BaseModel):
    url: str


# ══════════════════════════════════════════════════════════════════════════════
# MultiSourceRAG, VideoStore, WebpageStore
# ══════════════════════════════════════════════════════════════════════════════
from multi_source_rag import MultiSourceRAG
from document_loader import (
    ALLOWED_EXTENSIONS,
    FileValidationError,
    MAX_FILE_SIZE_BYTES,
    _get_whisper_model,
    load_url_advanced,
)
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
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Video upload failed")
        raise HTTPException(status_code=500, detail="Video ingestion failed unexpectedly.") from exc


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
            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract meaningful content from this URL. "
                    "The page may require JavaScript, a login, or block bots."
                ),
            )

        # ── Classify doc-level metadata (insurer, policy_type) ───────────────
        # This mirrors what _ingest_file() does for uploaded documents so that
        # webpage chunks participate in query-time metadata filtering just like
        # PDF/DOCX chunks do.
        from metadata_tagger import tag_document, classify_chunk_policy_type, classify_chunk_intent
        from router import get_insurance_llm

        llm = None
        try:
            llm = get_insurance_llm(temperature=0)
        except Exception as llm_exc:
            logger.warning("[upload-webpage] LLM unavailable — falling back to regex classification: %s", llm_exc)

        preview    = docs[0].page_content[:5000]
        extra_text = docs[0].page_content[5000:10000]
        page_title = docs[0].metadata.get("title", url)

        doc_meta = tag_document(
            page_title,
            preview,
            extra_text=extra_text,
            doc_type="general",
            llm=llm,
        )
        logger.info(
            "[upload-webpage] '%s' → insurer=%s, policy_type=%s",
            url, doc_meta.get("insurer", "UNKNOWN"), doc_meta.get("policy_type", "general"),
        )

        # ── Chunk ─────────────────────────────────────────────────────────────
        chunker = SectionChunker(chunk_size=600, chunk_overlap=80)
        chunks  = chunker.split_documents(docs)

        # ── Classify every chunk (section + policy_type) ─────────────────────
        for chunk in chunks:
            chunk.metadata["source_type"] = "webpage"
            chunk.metadata["source_url"]  = url

            # Section / intent classification
            chunk.metadata["section"] = classify_chunk_intent(
                chunk.page_content,
                doc_type="general",
                llm=llm,
                force_llm=False,
            )

            # Per-chunk policy_type — fall back to doc-level if still "general"
            chunk_policy = classify_chunk_policy_type(
                chunk.page_content,
                llm=llm,
                force_llm=False,
            )
            chunk.metadata["policy_type"] = (
                chunk_policy
                if chunk_policy != "general"
                else doc_meta.get("policy_type", "general")
            )

            # Propagate doc-level fields to every chunk
            chunk.metadata["insurer"]  = doc_meta.get("insurer", "UNKNOWN")
            chunk.metadata["doc_type"] = "general"
            chunk.metadata.setdefault("filename", url)
            chunk.metadata.setdefault("source",   url)

        assigned_policy_types = list({c.metadata.get("policy_type", "general") for c in chunks})
        assigned_sections     = list({c.metadata.get("section", "general")     for c in chunks})
        general_count = sum(1 for c in chunks if c.metadata.get("policy_type", "general") == "general")

        logger.info(
            "[upload-webpage] %d chunks | policy_types=%s | sections=%s | %d/%d still 'general'",
            len(chunks), assigned_policy_types, assigned_sections, general_count, len(chunks),
        )

        multi.add_webpage_chunks(url, chunks)
        return {
            "status":  "success",
            "url":     url,
            "title":   page_title,
            "chunks":  len(chunks),
            "insurer": doc_meta.get("insurer", "UNKNOWN"),
            "assigned_policy_types": assigned_policy_types,
            "assigned_sections":     assigned_sections,
            "chunks_still_general_policy_type": general_count,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Webpage upload failed")
        raise HTTPException(status_code=500, detail=f"Webpage ingestion failed: {exc}") from exc


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
        _conversation_agent = ConversationAgent(_get_pipeline().vector_store, _get_multi_rag())
        _conversation_agent.restore_sessions(_agent_sessions)
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
    history_list = await _get_conversation_history(req.session_id)
    history_str = ""
    for turn in history_list[-_MAX_HISTORY_TURNS * 2 :]:
        history_str += f"{turn['role'].capitalize()}: {turn['content']}\n"

    agent = _get_conversation_agent()
    try:
        result, is_complete = await asyncio.wait_for(
            agent.process_message(req.session_id, req.question, history_str),
            timeout=_ASK_TIMEOUT_SECONDS,
        )
        await _save_agent_sessions(agent.export_sessions())

        answer = result.get("message", "")
        options = result.get("options", [])  # list of {id, label, description, recommended}

        # Update conversation memory
        history_list.append({"role": "user", "content": req.question})
        history_list.append({"role": "assistant", "content": answer})
        await _save_conversation_history(req.session_id, history_list)

        return {
            "answer": answer,
            "options": options,
            "sources": [],
            "conversation_continues": not is_complete,
        }
    except asyncio.TimeoutError:
        logger.warning("Conversational ask timed out after %ds", _ASK_TIMEOUT_SECONDS)
        raise HTTPException(status_code=504, detail="The AI model server is taking too long to respond. Please try again in a moment.")
    except (APIConnectionError, APITimeoutError, APIStatusError) as exc:
        logger.warning("Conversational ask failed due to upstream model error: %s", exc)
        status_code, detail = _describe_llm_failure(exc)
        raise HTTPException(status_code=status_code, detail=detail) from exc
    except Exception as exc:
        logger.exception("Conversational ask failed")
        raise HTTPException(
            status_code=500,
            detail="The backend could not generate an answer due to an unexpected internal error.",
        ) from exc


@app.post("/conversation/reset/{session_id}")
async def reset_conversation(session_id: str):
    """Reset the conversational agent's state for a session (clears pending questions)."""
    agent = _get_conversation_agent()
    agent.reset_session(session_id)
    await _delete_agent_session(session_id)
    # Also clear the stored conversation history if desired
    await _delete_conversation_history(session_id)
    return {"status": "reset"}


@app.delete("/conversation/{session_id}")
async def clear_conversation(session_id: str):
    """Clear conversation history (legacy endpoint)."""
    await _delete_conversation_history(session_id)
    # Also reset agent state
    _get_conversation_agent().reset_session(session_id)
    await _delete_agent_session(session_id)
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
            timeout=_ASK_TIMEOUT_SECONDS,
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
        raise HTTPException(
            status_code=500,
            detail="The backend could not generate an answer due to an unexpected internal error.",
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# ALL ORIGINAL ENDPOINTS REMAIN UNCHANGED BELOW
# ══════════════════════════════════════════════════════════════════════════════

# ── Upload (async) ────────────────────────────────────────────────────────────
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    await _prune_jobs()
    suffix = os.path.splitext(file.filename or "")[1].lower()
    supported = ALLOWED_EXTENSIONS
    if suffix not in supported:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")
    filename = file.filename or f"upload{suffix}"
    tmp_path = await _save_upload_to_temp(file, suffix, filename)
    job_id = str(uuid.uuid4())
    await _set_job(job_id, _job_state("queued", filename))

    async def _process():
        try:
            async with _get_ingest_semaphore():
                await _set_job(job_id, _job_state("processing", filename))
                chunks = await asyncio.to_thread(_ingest_file, tmp_path, filename)
                await _set_job(job_id, _job_state("done", filename, chunks=chunks))
                logger.info("Ingested %s — %d chunks", filename, chunks)
        except FileValidationError as exc:
            await _set_job(job_id, _job_state("error", filename, error=str(exc)))
            logger.warning("File validation failed for %s: %s", filename, exc)
        except Exception as exc:
            await _set_job(job_id, _job_state("error", filename, error="Document ingestion failed unexpectedly."))
            logger.exception("Ingest failed for %s", filename)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    asyncio.create_task(_process())
    return {"job_id": job_id, "filename": filename, "status": "queued"}


@app.get("/upload/{job_id}")
async def upload_status(job_id: str):
    job = await _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


# ── Ingest URL (async) ────────────────────────────────────────────────────────
@app.post("/ingest-url")
async def ingest_url(req: URLRequest):
    await _prune_jobs()
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="Invalid URL — must start with http:// or https://")
    job_id = str(uuid.uuid4())
    await _set_job(job_id, _job_state("queued", url))

    async def _process():
        try:
            async with _get_ingest_semaphore():
                await _set_job(job_id, _job_state("processing", url))
                chunks = await asyncio.to_thread(_get_pipeline().add_url, url)
                await _set_job(job_id, _job_state("done", url, chunks=chunks))
                logger.info("Ingested URL %s — %d chunks", url, chunks)
        except Exception:
            await _set_job(job_id, _job_state("error", url, error="URL ingestion failed unexpectedly."))
            logger.exception("URL ingest failed for %s", url)

    asyncio.create_task(_process())
    return {"job_id": job_id, "url": url, "status": "queued"}


# ── Ask with URL (ultra‑fast) ────────────────────────────────────────────────
class AskURLRequest(BaseModel):
    url: str
    question: str


async def fetch_url_text_async(url: str, max_chars: int = 1500) -> str:
    """
    Async helper used by /ask-url for fast, lightweight URL fetching.
    Uses the same tag-removal and noise-filtering logic as the main
    document_loader._load_web_page() so results are consistent.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "Mozilla/5.0 (compatible; InsureHubBot/1.0)"},
            ) as resp:
                resp.raise_for_status()
                html = await resp.text(errors="replace")

        soup = BeautifulSoup(html, "html.parser")

        # Remove the same boilerplate tags as document_loader
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "noscript", "svg", "iframe",
                         "button", "figure", "picture"]):
            tag.decompose()

        container = (
            soup.find("main")
            or soup.find("article")
            or soup.find("section")
            or soup.find("body")
            or soup
        )
        raw_text = container.get_text(separator="\n", strip=True)

        # Apply noise filter: drop lines shorter than 4 words
        lines = [l for l in raw_text.splitlines() if len(l.strip().split()) >= 4]
        text  = "\n".join(lines)
        text  = re.sub(r"\n{3,}", "\n\n", text).strip()

        return text[:max_chars] if text else ""
    except Exception as exc:
        logger.error("URL fetch error for %s: %s", url, exc)
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
        from router import VLLM_API_KEY, VLLM_HOST, VLLM_MODEL
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage

        llm = ChatOpenAI(
            model=VLLM_MODEL,
            base_url=f"{VLLM_HOST}/v1",
            api_key=VLLM_API_KEY,
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
            yield f"data: {_json.dumps({'error': 'Unexpected server error.'})}\n\n".encode()

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── Streaming ask endpoint (documents only, unchanged) ────────────────────────
@app.post("/ask-stream")
async def ask_stream(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    async def generate():
        pipeline = _get_pipeline()
        try:
            answer, _, _ = await asyncio.wait_for(
                asyncio.to_thread(pipeline.knowledge_query, req.question),
                timeout=_ASK_TIMEOUT_SECONDS,
            )
            for chunk in [answer[i:i+30] for i in range(0, len(answer), 30)]:
                yield chunk
                await asyncio.sleep(0.01)
        except asyncio.TimeoutError:
            logger.warning("Streaming ask timed out after %ds", _ASK_TIMEOUT_SECONDS)
            yield "Error: The AI model server is taking too long to respond. Please try again in a moment."
        except Exception as exc:
            logger.exception("Streaming ask failed")
            yield "Error: The backend could not generate an answer due to an unexpected internal error."

    return StreamingResponse(generate(), media_type="text/plain")


# ── Docs management ───────────────────────────────────────────────────────────
@app.get("/docs")
async def list_docs():
    pipeline = _get_pipeline()
    filenames = pipeline.vector_store.list_values("filename")
    counts = {}
    try:
        all_meta = pipeline.vector_store.collection.get(include=["metadatas"])
        for meta in all_meta["metadatas"]:
            fn = meta.get("filename")
            if fn:
                counts[fn] = counts.get(fn, 0) + 1
    except Exception:
        pass
    documents = [f"{s} ({counts.get(s, 0)} chunks)" for s in filenames]
    return {"documents": documents}


@app.delete("/docs")
async def clear_docs():
    await asyncio.to_thread(_get_pipeline().clear_documents)
    return {"status": "cleared"}


@app.delete("/docs/{name:path}")
async def remove_doc(name: str):
    pipeline = _get_pipeline()
    filenames_before = set(pipeline.vector_store.list_values("filename"))
    if name not in filenames_before:
        raise HTTPException(status_code=404, detail=f"Document '{name}' not found.")
    pipeline.vector_store.delete_by_field("filename", name)
    return {"removed": True, "filename": name}


# ── Voice transcription (Whisper) ────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "recording.webm")[1].lower()
    if suffix not in {".webm", ".wav", ".mp3", ".m4a"}:
        raise HTTPException(status_code=400, detail="Unsupported audio format. Use webm, wav, mp3, or m4a.")
    tmp_path = await _save_upload_to_temp(file, suffix, file.filename or f"recording{suffix}")
    try:
        model = await asyncio.to_thread(_get_whisper_model)
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
    return {"status": "ok", "chunks": pipeline.vector_store.count()}


# ══════════════════════════════════════════════════════════════════════════════
# RAG EVALUATION ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

class RetrieveRequest(BaseModel):
    question: str
    top_k: int = 12


class EvaluateRequest(BaseModel):
    question: str
    top_k: int = 12
    run_ragas: bool = True


def _serialize_chunk(rank: int, doc) -> dict:
    """Convert a LangChain Document to a JSON-serialisable dict."""
    meta = dict(doc.metadata) if doc.metadata else {}
    # Convert any non-serialisable values (e.g. keyword lists) to strings
    for k, v in list(meta.items()):
        if isinstance(v, (list, dict)):
            meta[k] = _json.dumps(v)
        elif not isinstance(v, (str, int, float, bool, type(None))):
            meta[k] = str(v)
    return {
        "rank": rank,
        "content": doc.page_content,
        "metadata": {
            "filename":    meta.get("filename", meta.get("source", "Unknown")),
            "source":      meta.get("source", "Unknown"),
            "page":        meta.get("page", "?"),
            "section":     meta.get("section", "general"),
            "insurer":     meta.get("insurer", "—"),
            "policy_type": meta.get("policy_type", "—"),
            "similarity":  round(float(meta.get("similarity", 0)), 4),
            "rerank_score": round(float(meta.get("rerank_score", meta.get("similarity", 0))), 4),
            "keywords":    meta.get("keywords", ""),
        },
    }


# ── /retrieve — pure retrieval, NO LLM call ──────────────────────────────────
@app.post("/retrieve")
async def retrieve_chunks(req: RetrieveRequest):
    """
    Return ranked chunks + full metadata for a query.
    No LLM is called — this is pure vector/hybrid retrieval.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if _get_pipeline().vector_store.count() == 0:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Upload a document first.")

    t0 = time.time()
    try:
        chunks = await asyncio.to_thread(
            _get_pipeline().vector_store.search,
            req.question,
            req.top_k,
            None,          # filter_metadata
            True,          # use_hybrid
            True,          # use_reranker
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    retrieval_ms = round((time.time() - t0) * 1000)
    serialised = [_serialize_chunk(i + 1, doc) for i, doc in enumerate(chunks)]
    return {
        "chunks": serialised,
        "total_chunks": len(serialised),
        "retrieval_time_ms": retrieval_ms,
    }


# ── /evaluate — full pipeline evaluation with optional RAGAS ─────────────────
@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    """
    Full RAG evaluation pipeline:
      1. Retrieve chunks  → retrieval_time_ms
      2. Generate answer  → llm_time_ms
      3. RAGAS scoring    → ragas_time_ms  (skipped when run_ragas=false)
    Returns chunks, metadata, answer, timings, and RAGAS scores.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if _get_pipeline().vector_store.count() == 0:
        raise HTTPException(status_code=400, detail="Knowledge base is empty. Upload a document first.")

    # ── Step 1: Retrieve ──────────────────────────────────────────────────────
    t_retrieve_start = time.time()
    try:
        chunks = await asyncio.to_thread(
            _get_pipeline().vector_store.search,
            req.question,
            req.top_k,
            None, True, True,
        )
    except Exception as exc:
        logger.exception("Retrieval failed during evaluation")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc
    retrieval_ms = round((time.time() - t_retrieve_start) * 1000)

    serialised_chunks = [_serialize_chunk(i + 1, doc) for i, doc in enumerate(chunks)]
    contexts = [doc.page_content for doc in chunks]

    # ── Step 2: Generate LLM answer ───────────────────────────────────────────
    t_llm_start = time.time()
    answer = ""
    sources = []
    try:
        answer, _, sources = await asyncio.wait_for(
            asyncio.to_thread(_get_pipeline().knowledge_query, req.question),
            timeout=_ASK_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        answer = "⚠️ LLM timed out."
    except Exception as exc:
        logger.warning("LLM generation failed during evaluation: %s", exc)
        answer = f"⚠️ LLM error: {exc}"
    llm_ms = round((time.time() - t_llm_start) * 1000)

    # ── Step 3: RAGAS scoring ─────────────────────────────────────────────────
    ragas_ms = 0
    ragas_scores: dict = {
        "faithfulness": None,
        "answer_relevancy": None,
        "context_precision": None,
        "error": None,
    }

    if req.run_ragas and contexts and answer and not answer.startswith("⚠️"):
        t_ragas_start = time.time()
        try:
            from datasets import Dataset
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_community.embeddings import HuggingFaceEmbeddings

            from router import VLLM_HOST, VLLM_MODEL, VLLM_API_KEY

            ragas_llm = ChatOpenAI(
                model=VLLM_MODEL,
                base_url=f"{VLLM_HOST}/v1",
                api_key=VLLM_API_KEY,
                temperature=0,
                max_tokens=512,
                timeout=60,
                max_retries=1,
            )
            ragas_llm_wrapper = LangchainLLMWrapper(ragas_llm)

            embed_model = _get_pipeline().vector_store.embed_model
            ragas_embed_wrapper = LangchainEmbeddingsWrapper(embed_model)

            eval_dataset = Dataset.from_dict({
                "question":  [req.question],
                "answer":    [answer],
                "contexts":  [contexts[:6]],   # cap to 6 to stay within token limits
                "ground_truth": [""],           # empty — not required for these 3 metrics
            })

            metrics_to_run = [faithfulness, answer_relevancy, context_precision]
            for m in metrics_to_run:
                m.llm = ragas_llm_wrapper
                if hasattr(m, "embeddings"):
                    m.embeddings = ragas_embed_wrapper

            result = await asyncio.wait_for(
                asyncio.to_thread(ragas_evaluate, eval_dataset, metrics=metrics_to_run),
                timeout=120,
            )
            result_dict = result.to_pandas().iloc[0].to_dict()
            ragas_scores["faithfulness"]       = round(float(result_dict.get("faithfulness", 0) or 0), 4)
            ragas_scores["answer_relevancy"]   = round(float(result_dict.get("answer_relevancy", 0) or 0), 4)
            ragas_scores["context_precision"]  = round(float(result_dict.get("context_precision", 0) or 0), 4)
        except asyncio.TimeoutError:
            ragas_scores["error"] = "RAGAS evaluation timed out after 120s."
            logger.warning("RAGAS timed out")
        except ImportError as exc:
            ragas_scores["error"] = f"RAGAS library not installed: {exc}"
            logger.warning("RAGAS import failed: %s", exc)
        except Exception as exc:
            ragas_scores["error"] = f"RAGAS evaluation failed: {exc}"
            logger.warning("RAGAS failed: %s", exc)
        finally:
            ragas_ms = round((time.time() - t_ragas_start) * 1000)

    total_ms = retrieval_ms + llm_ms + ragas_ms
    return {
        "answer":  answer,
        "sources": sources,
        "chunks":  serialised_chunks,
        "timing": {
            "retrieval_ms": retrieval_ms,
            "llm_ms":       llm_ms,
            "ragas_ms":     ragas_ms,
            "total_ms":     total_ms,
        },
        "ragas": ragas_scores,
    }