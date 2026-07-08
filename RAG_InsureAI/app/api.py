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
from urllib.parse import unquote
from bs4 import BeautifulSoup

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from auth import create_login_endpoint, require_auth
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from agent_hub import hub as _agent_hub
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
    "http://localhost:4000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:8001",
    "http://localhost:8002",
    "http://localhost:8501",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8001",
    "http://127.0.0.1:8002",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:8080",
    "https://insureai-chat.lovable.app",
    "https://id-preview--1f3edfb5-f351-48b6-baff-3a69cba3ed88.lovable.app",
    "https://insurehub-rag-frontend-zqp6.vercel.app",
    "https://insurehub-your-ai-insurance-advisor.vercel.app",
    # Admin + Agent panels on Vercel — update these after deploying panels/
    "https://insurehub-panels.vercel.app",
    # Direct IP access
    "http://123.253.124.14:8501",
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
create_login_endpoint(app)

from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_FRONTEND_DIST = "/app/frontend_dist"

# Serve the React app's static assets (/assets/*, /favicon.ico, etc.)
if os.path.isdir(_FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(_FRONTEND_DIST, "assets")), name="assets")

@app.get("/")
async def root():
    """Serve the React chat frontend if built, otherwise redirect to /auth."""
    _index = os.path.join(_FRONTEND_DIST, "index.html")
    if os.path.isfile(_index):
        return FileResponse(_index)
    return RedirectResponse(url="/auth", status_code=302)

@app.get("/auth")
async def auth_page():
    return FileResponse(os.path.join(_FRONTEND_DIST, "auth.html"))

@app.get("/admin")
async def admin_page():
    return FileResponse(os.path.join(_FRONTEND_DIST, "admin.html"))

@app.get("/agent-dashboard")
async def agent_dashboard_page():
    return FileResponse(os.path.join(_FRONTEND_DIST, "agent-dashboard.html"))

@app.get("/super-admin")
async def super_admin_page():
    return FileResponse(os.path.join(_APP_DIR, "super_admin.html"))


@app.get("/tunnel-url")
async def tunnel_url_endpoint():
    """Return the current Cloudflare tunnel URL (read from file written by tunnel_watcher.py)."""
    import urllib.request as _urllib
    import re as _re
    # Try live metrics first (most accurate)
    try:
        with _urllib.urlopen("http://localhost:20241/metrics", timeout=2) as r:
            text = r.read().decode()
        m = _re.search(r'userHostname="(https://[^"]+trycloudflare\.com)"', text)
        if m:
            return {"url": m.group(1), "source": "live"}
    except Exception:
        pass
    # Fall back to file written by tunnel_watcher.py (inside mounted app/ dir)
    url_file = os.path.join(_APP_DIR, "tunnel_url.txt")
    if os.path.exists(url_file):
        try:
            with open(url_file) as f:
                url = f.read().strip()
            if url:
                return {"url": url, "source": "file"}
        except OSError:
            pass
    return {"url": None, "source": "none"}


# ── Human handoff session endpoints ──────────────────────────────────────────

_ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "insurehub2026")
_SUPER_ADMIN_PASSWORD = os.getenv("SUPER_ADMIN_PASSWORD", "superadmin2026")
_AGENT_PASSWORD = os.getenv("AGENT_PASSWORD", "") or _ADMIN_PASSWORD


# ── Super-admin API ────────────────────────────────────────────────────────────

def _check_super_admin(x_super_admin_token: str = Header(None)):
    if not x_super_admin_token or x_super_admin_token not in _agent_hub._super_admin_tokens:
        raise HTTPException(status_code=403, detail="Not authorized")
    return x_super_admin_token


@app.post("/super-admin/login")
async def super_admin_login(request: Request):
    data = await request.json()
    if data.get("password") != _SUPER_ADMIN_PASSWORD:
        raise HTTPException(status_code=401, detail="Invalid password")
    token = uuid.uuid4().hex
    _agent_hub._super_admin_tokens.add(token)
    return {"token": token}


@app.get("/super-admin/data")
async def super_admin_data(token: str = Depends(_check_super_admin)):
    return _agent_hub.get_super_admin_data()


@app.get("/super-admin/agent/{name}/chats")
async def super_admin_agent_chats(name: str, token: str = Depends(_check_super_admin)):
    rec = _agent_hub._agent_records.get(name)
    if not rec:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"name": name, "chats": rec.get("chats", [])}


@app.get("/super-admin/sessions")
async def super_admin_all_sessions(token: str = Depends(_check_super_admin)):
    return {"sessions": _agent_hub.get_all_sessions_for_super_admin()}


@app.get("/super-admin/session/{session_id}/messages")
async def super_admin_session_messages(session_id: str, token: str = Depends(_check_super_admin)):
    msgs = _agent_hub.get_session_full_messages(session_id)
    return {"session_id": session_id, "messages": msgs}


@app.post("/super-admin/agent/{name}/block")
async def super_admin_block(name: str, token: str = Depends(_check_super_admin)):
    ok = _agent_hub.block_agent(name)
    await _agent_hub._broadcast_super_admin_update()
    await _agent_hub._broadcast_sessions_update()
    return {"ok": ok}


@app.post("/super-admin/agent/{name}/unblock")
async def super_admin_unblock(name: str, token: str = Depends(_check_super_admin)):
    ok = _agent_hub.unblock_agent(name)
    await _agent_hub._broadcast_super_admin_update()
    await _agent_hub._broadcast_sessions_update()
    return {"ok": ok}


@app.post("/super-admin/assign")
async def super_admin_assign(request: Request, token: str = Depends(_check_super_admin)):
    data = await request.json()
    ok = await _agent_hub.super_admin_assign_session(
        data.get("session_id", ""), data.get("agent_name", "")
    )
    return {"ok": ok}


@app.delete("/super-admin/session/{session_id}")
async def super_admin_delete_session(session_id: str, token: str = Depends(_check_super_admin)):
    ok = await _agent_hub.delete_session(session_id)
    return {"ok": ok}


class BackendSettingsRequest(BaseModel):
    mode: str                      # "auto" | "vllm" | "groq" | "manual"
    manual_api_key: str = ""       # only sent when the admin is setting/changing it
    manual_base_url: str = ""
    manual_model: str = ""


@app.get("/super-admin/backend-settings")
async def super_admin_get_backend_settings(token: str = Depends(_check_super_admin)):
    from router import get_backend_settings
    return get_backend_settings()


@app.post("/super-admin/backend-settings")
async def super_admin_set_backend_settings(req: BackendSettingsRequest, token: str = Depends(_check_super_admin)):
    from router import set_backend_settings
    try:
        return set_backend_settings(
            mode=req.mode,
            manual_api_key=req.manual_api_key,
            manual_base_url=req.manual_base_url,
            manual_model=req.manual_model,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.websocket("/ws/super-admin")
async def ws_super_admin(websocket: WebSocket):
    await websocket.accept()
    try:
        auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=15)
        if auth_msg.get("token") not in _agent_hub._super_admin_tokens:
            await websocket.send_json({"type": "error", "message": "Unauthorized"})
            await websocket.close(code=1008)
            return
        _agent_hub.register_super_admin(websocket)
        await websocket.send_json({"type": "connected", **_agent_hub.get_super_admin_data()})
        while True:
            msg = await websocket.receive_json()
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception:
        logger.exception("Super-admin WebSocket error")
    finally:
        _agent_hub.unregister_super_admin(websocket)


@app.post("/session/create")
async def session_create():
    """Create a monitored chat session for human handoff tracking."""
    sid = _agent_hub.create_session()
    return {"session_id": sid}


@app.get("/agents/status")
async def agents_status():
    """Returns how many human agents are currently online."""
    return {"online": _agent_hub.online_count()}


@app.delete("/session/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Delete a session (triggered from agent dashboard)."""
    deleted = await _agent_hub.delete_session(session_id)
    return {"deleted": deleted}


@app.post("/session/{session_id}/request-handoff")
async def session_request_handoff(session_id: str):
    """HTTP fallback for handoff when the user WebSocket has dropped."""
    assigned = await _agent_hub.request_handoff(session_id)
    session = _agent_hub.get_session(session_id)
    return {"assigned": assigned, "status": session.status if session else "unknown"}


@app.post("/session/{session_id}/cancel-handoff")
async def session_cancel_handoff(session_id: str):
    """HTTP fallback for cancelling a pending handoff when the user WebSocket
    has dropped — without this, cancelHandoff() on the frontend had no way to
    tell the server, so the session stayed "waiting" and the next poll tick
    flipped the UI right back, making Cancel look completely unresponsive."""
    cancelled = await _agent_hub.cancel_handoff(session_id)
    session = _agent_hub.get_session(session_id)
    return {"cancelled": cancelled, "status": session.status if session else "unknown"}


@app.get("/session/{session_id}/poll")
async def session_poll(session_id: str, after: int = 0):
    """
    Polling fallback for human-agent messages.
    Returns session status + all messages after the given index.
    Used when WebSocket is unavailable (e.g. Cloudflare timeout).
    """
    session = _agent_hub.get_session(session_id)
    if not session:
        return {"status": "ai", "agent_name": "", "messages": [], "total": 0}
    msgs = []
    for m in session.history[after:]:
        entry = {"role": m.role, "content": m.content, "timestamp": m.timestamp}
        # An agent reply from the "Unanswered Queries" panel answers a specific
        # earlier question (meta.answers_index), not necessarily the user's most
        # recent message — surface that question's text so the widget can show
        # the user which of their (possibly several) questions this is replying to.
        answers_index = m.meta.get("answers_index") if m.role == "agent" else None
        if answers_index is not None and 0 <= answers_index < len(session.history):
            entry["answers_question"] = session.history[answers_index].content
        msgs.append(entry)
    agent_name = ""
    if session.agent_id and session.agent_id in _agent_hub._agents:
        agent_name = _agent_hub._agents[session.agent_id].name
    return {
        "status": session.status,
        "agent_name": agent_name,
        "messages": msgs,
        "total": len(session.history),
    }


@app.websocket("/ws/agent")
async def ws_agent_endpoint(websocket: WebSocket):
    """WebSocket for human agents — login, monitor sessions, take over chats."""
    await websocket.accept()
    agent_id: str | None = None
    try:
        data = await asyncio.wait_for(websocket.receive_json(), timeout=30)
        if data.get("type") != "login":
            await websocket.close(code=1008)
            return
        name = (data.get("name") or "Agent").strip() or "Agent"
        if data.get("password") != _AGENT_PASSWORD:
            await websocket.send_json({"type": "error", "message": "Invalid password."})
            await websocket.close(code=1008)
            return
        agent_id = uuid.uuid4().hex[:8]
        _agent_hub.register_agent(agent_id, name, websocket)
        await websocket.send_json({"type": "logged_in", "agent_id": agent_id, "name": name})
        await websocket.send_json({"type": "sessions_update", "sessions": _agent_hub.list_sessions()})
        await _agent_hub._broadcast_super_admin_update()
        while True:
            msg = await websocket.receive_json()
            t = msg.get("type", "")
            if t == "monitor":
                await _agent_hub.agent_monitor(agent_id, msg.get("session_id", ""))
            elif t == "takeover":
                await _agent_hub.agent_takeover(agent_id, msg.get("session_id", ""))
                await _agent_hub._broadcast_super_admin_update()
            elif t == "message":
                await _agent_hub.agent_send_message(agent_id, msg.get("session_id", ""), msg.get("content", ""))
            elif t == "answer_unanswered":
                await _agent_hub.agent_answer_unanswered(
                    agent_id, msg.get("session_id", ""), msg.get("index", -1), msg.get("content", "")
                )
            elif t == "release":
                await _agent_hub.agent_release(agent_id)
            elif t == "accept_handoff":
                await _agent_hub.accept_handoff(agent_id, msg.get("session_id", ""))
                await _agent_hub._broadcast_super_admin_update()
            elif t == "decline_handoff":
                await _agent_hub.decline_handoff(agent_id, msg.get("session_id", ""))
    except (WebSocketDisconnect, asyncio.TimeoutError):
        pass
    except Exception:
        logger.exception("Agent WebSocket error")
    finally:
        if agent_id:
            await _agent_hub.unregister_agent(agent_id)


@app.websocket("/ws/user/{session_id}")
async def ws_user_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket for users — always open as a notification channel.
    Handoff is triggered explicitly by the client sending {"type":"request_handoff"},
    not automatically on connect.
    """
    session = _agent_hub.get_or_create_session(session_id)
    await websocket.accept()
    session.user_ws = websocket

    # Deliver any buffered message that was produced while the WS was disconnected
    if session.pending_ws_message:
        try:
            await websocket.send_json(session.pending_ws_message)
        except Exception:
            pass
        session.pending_ws_message = None

    try:
        while True:
            msg = await websocket.receive_json()
            msg_type = msg.get("type", "")
            if msg_type == "request_handoff":
                # Fast-path: if handoff already resolved this turn, tell the frontend immediately
                if session.handoff_exhausted:
                    await websocket.send_json({
                        "type": "handoff_timeout",
                        "message": "No agent is available right now. We've emailed our support team — someone will reach out to you soon!",
                    })
                elif session.status == "human":
                    # Agent already joined before WS connected
                    agent_name = ""
                    if session.agent_id and session.agent_id in _agent_hub._agents:
                        agent_name = _agent_hub._agents[session.agent_id].name
                    await websocket.send_json({"type": "agent_joined", "agent_name": agent_name})
                else:
                    notified = await _agent_hub.request_handoff(session_id)
                    if notified:
                        await websocket.send_json({
                            "type": "waiting",
                            "message": "Hang tight! An agent has been notified and will join shortly...",
                        })
                    else:
                        # No agents available (all offline or all declined this session)
                        unanswerable = next(
                            (m.content for m in reversed(session.history) if m.role == "user"), ""
                        )
                        asyncio.create_task(
                            _agent_hub.trigger_offline_escalation(session_id, unanswerable)
                        )
                        await websocket.send_json({
                            "type": "handoff_timeout",
                            "message": "No agent is available right now. We've emailed our support team — someone will reach out to you soon!",
                        })
            elif msg_type == "cancel_handoff":
                # User explicitly cancelled waiting — send email and release back to AI
                cancelled = await _agent_hub.cancel_handoff(session_id)
                if cancelled:
                    await websocket.send_json({
                        "type": "handoff_timeout",
                        "message": "No problem! I've sent your question to our support team — someone will follow up with you soon. You can keep chatting with me in the meantime! 😊",
                    })
            elif msg_type == "message":
                content = (msg.get("content") or "").strip()
                if content and session.status == "human":
                    await _agent_hub.user_message_to_agent(session_id, content)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        session.user_ws = None

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
@app.on_event("startup")
async def _warmup_pipeline():
    """Pre-load embedding model and vector store at startup so first request is fast."""
    await asyncio.to_thread(_get_pipeline)
    logger.info("Pipeline warmed up — embedding model loaded and ready.")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_csv_env("CORS_ALLOW_ORIGINS", _DEFAULT_CORS_ORIGINS),
    allow_origin_regex=r"https://.*\.(lovable\.app|lovableproject\.com|vercel\.app)",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
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

    # ── Step 2: Tag document — use regex only (llm=None) so ingest never
    # blocks on slow LLM calls. Regex tagging is fast and reliable enough
    # for metadata; LLM refinement can take 60-120s and causes job timeouts.
    llm = None
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
    doc_meta = {"filename": filename, "doc_type": doc_type, **doc_tags}
    # Run summary generation in a daemon thread so it doesn't block the job
    # from completing. The summary is optional — retrieval degrades gracefully
    # without it, so a failure or slow LLM response is not critical.
    import threading as _threading
    _threading.Thread(
        target=pipeline._upsert_summary,
        args=(chunks, unique_source, doc_meta, llm),
        daemon=True,
    ).start()
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


def _chunk_transcript(transcript_text: str, url: str, title: str = "", doc_meta: dict | None = None) -> list:
    from langchain_core.documents import Document

    if doc_meta is None:
        doc_meta = {}

    chunker = SectionChunker(chunk_size=500, chunk_overlap=100)
    doc = Document(
        page_content=transcript_text,
        metadata={
            "source_url":  url,
            "source":      url,       # needed for is_youtube check in SectionChunker
            "title":       title,
            "doc_type":    "youtube", # tells SemanticChunker to use word-window splitting
            "source_type": "youtube_transcript",
        },
    )
    chunks = chunker.split_documents([doc], doc_type="youtube")
    for chunk in chunks:
        # source_type="video" so VideoVectorStore.search(filter={"source_type":"video"}) matches
        chunk.metadata["source_type"] = "video"
        chunk.metadata["source_url"]  = url
        # Propagate doc-level metadata so insurer/policy_type filters work on video chunks
        chunk.metadata.setdefault("insurer",     doc_meta.get("insurer",     "UNKNOWN"))
        chunk.metadata.setdefault("policy_type", doc_meta.get("policy_type", "general"))
        chunk.metadata.setdefault("section",     doc_meta.get("section",     "general"))
        chunk.metadata.setdefault("keywords",    doc_meta.get("keywords",    ""))
    return chunks


# ── Upload Video (any video URL) ───────────────────────────────────────────────────
@app.post("/upload-video")
async def upload_video(req: URLRequest, _: str = Depends(require_auth)):
    from video_store import _normalize_video_url
    url = _normalize_video_url(req.url.strip())
    multi = _get_multi_rag()
    if multi.video_exists(url):
        return {"status": "already_exists", "url": url, "message": "Video already in knowledge base."}
    try:
        from document_loader import load_url
        from metadata_tagger import tag_document
        from router import get_insurance_llm

        docs = await asyncio.to_thread(load_url, url)
        if not docs or not any(d.page_content.strip() for d in docs):
            raise HTTPException(status_code=400, detail="Could not extract transcript from this video.")
        transcript_text = " ".join(d.page_content for d in docs if d.page_content.strip())
        title = docs[0].metadata.get("video_title") or docs[0].metadata.get("title") or url

        # Classify insurer / policy_type so filters work on video chunks
        llm = None
        try:
            llm = get_insurance_llm(temperature=0)
        except Exception as llm_exc:
            logger.warning("[upload-video] LLM unavailable — falling back to regex classification: %s", llm_exc)

        preview    = transcript_text[:5000]
        extra_text = transcript_text[5000:10000]
        doc_meta = tag_document(title, preview, extra_text=extra_text, doc_type="youtube", llm=llm)
        logger.info(
            "[upload-video] '%s' → insurer=%s, policy_type=%s",
            url, doc_meta.get("insurer", "UNKNOWN"), doc_meta.get("policy_type", "general"),
        )

        chunks = _chunk_transcript(transcript_text, url, title, doc_meta=doc_meta)
        multi.add_video_chunks(url, chunks, title=title)
        return {
            "status":  "success",
            "url":     url,
            "title":   title,
            "chunks":  len(chunks),
            "insurer": doc_meta.get("insurer", "UNKNOWN"),
            "policy_type": doc_meta.get("policy_type", "general"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Video upload failed")
        msg = str(exc)
        if "Video unavailable" in msg or "unavailable" in msg.lower():
            raise HTTPException(status_code=400, detail="Video is unavailable or private. Please check the URL and try again.") from exc
        if "no subtitles" in msg.lower() or "transcript" in msg.lower():
            raise HTTPException(status_code=400, detail="No transcript found for this video. Try a video with captions enabled.") from exc
        raise HTTPException(status_code=500, detail=f"Video ingestion failed: {msg[:200]}") from exc


# ── Upload Webpage (permanent) ───────────────────────────────────────────────
@app.post("/upload-webpage")
async def upload_webpage(req: URLRequest, _: str = Depends(require_auth)):
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
        chunker = SectionChunker(chunk_size=2000, chunk_overlap=600)
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
    results = []
    seen_urls = set()

    # Videos uploaded via /upload-video — titles stored in video_store metadata
    for item in multi.video_store.list_videos_with_titles():
        if item["url"] not in seen_urls:
            seen_urls.add(item["url"])
            results.append(item)

    # Also surface YouTube transcripts stored as docs (uploaded via file upload)
    pipeline = _get_pipeline()
    try:
        for fn in pipeline.vector_store.list_values("filename"):
            if fn.startswith("youtube_"):
                rest = fn[len("youtube_"):]
                parts = rest.split("_", 1)
                video_id = parts[0]
                yt_url = f"https://www.youtube.com/watch?v={video_id}"
                if yt_url not in seen_urls:
                    seen_urls.add(yt_url)
                    raw_title = parts[1].replace("_", " ").replace(".txt", "").strip() if len(parts) > 1 else yt_url
                    results.append({"url": yt_url, "title": raw_title})
    except Exception:
        pass

    return {"videos": results}


# ── Delete video ─────────────────────────────────────────────────────────────
@app.delete("/videos")
async def delete_video(url: str, _: str = Depends(require_auth)):
    from video_store import _normalize_video_url
    import re as _re
    url = _normalize_video_url(unquote(url))
    multi = _get_multi_rag()
    deleted = False

    # Source 1: videos added via /upload-video (stored in video_store)
    if multi.video_exists(url):
        multi.delete_video(url)
        deleted = True

    # Source 2: YouTube transcripts uploaded as files (stored in pipeline doc store
    # with filename like "youtube_{video_id}_Title.txt")
    m = _re.search(r"[?&]v=([^&]+)", url) or _re.search(r"youtu\.be/([^?]+)", url)
    if m:
        video_id = m.group(1)
        pipeline = _get_pipeline()
        try:
            for fn in list(pipeline.vector_store.list_values("filename")):
                if fn.startswith(f"youtube_{video_id}"):
                    pipeline.vector_store.delete_by_field("filename", fn)
                    deleted = True
        except Exception as exc:
            logger.warning("Failed to delete pipeline video chunks for %s: %s", url, exc)

    if not deleted:
        raise HTTPException(status_code=404, detail="Video URL not found.")
    return {"removed": True, "url": url}


# ── List webpages ────────────────────────────────────────────────────────────
@app.get("/webpages")
async def list_webpages():
    multi = _get_multi_rag()
    return {"webpages": multi.list_webpages()}


# ── Delete webpage ───────────────────────────────────────────────────────────
@app.delete("/webpages")
async def delete_webpage(url: str, _: str = Depends(require_auth)):
    url = unquote(url)
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
        needs_human = result.get("needs_human", False)

        # Update conversation memory
        history_list.append({"role": "user", "content": req.question})
        history_list.append({"role": "assistant", "content": answer})
        await _save_conversation_history(req.session_id, history_list)

        # ── Log to agent hub + trigger handoff if needed ─────────────────────
        if req.session_id and req.session_id.strip():
            _agent_hub.get_or_create_session(req.session_id)
            agents_online = _agent_hub.online_count() > 0
            offline_escalated = needs_human and not agents_online

            if offline_escalated:
                # Log first so the history is complete before the email is sent
                await _agent_hub.log_message(req.session_id, "user", req.question)
                await _agent_hub.log_message(req.session_id, "ai", answer)
                asyncio.create_task(
                    _agent_hub.trigger_offline_escalation(req.session_id, req.question)
                )
            else:
                asyncio.create_task(_agent_hub.log_message(req.session_id, "user", req.question))
                asyncio.create_task(_agent_hub.log_message(req.session_id, "ai", answer))
                if needs_human and agents_online:
                    # Fire the popup on the agent dashboard — pass question directly
                    # to avoid race where history isn't written yet
                    asyncio.create_task(_agent_hub.request_handoff(req.session_id, req.question))

        return {
            "answer": answer,
            "options": options,
            "sources": [],
            "conversation_continues": not is_complete,
            "needs_human": needs_human,
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
async def upload(file: UploadFile = File(...), _: str = Depends(require_auth)):
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
        from router import get_insurance_llm
        from langchain_core.messages import HumanMessage

        llm = get_insurance_llm(temperature=0.3, max_tokens=200)
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
_STREAM_GREETINGS = {
    "hi", "hello", "hey", "hiya", "heya", "yo", "sup", "howdy",
    "hi there", "hello there", "hey there", "hiya there",
    "good morning", "good afternoon", "good evening", "good night",
    "gm", "gn", "good day",
    "how are you", "how are you doing", "how are you today",
    "how's it going", "how is it going", "how's everything",
    "how are things", "how do you do", "how have you been",
    "how's your day", "you doing okay", "you good", "all good",
    "thanks", "thank you", "thank you so much", "thanks a lot",
    "thanks so much", "many thanks", "ty", "thx", "thnx", "cheers",
    "much appreciated", "appreciate it", "thank u", "thanks mate",
    "bye", "goodbye", "good bye", "bye bye", "later", "see ya",
    "see you", "see you later", "take care", "cya", "ttyl",
    "have a good day", "have a great day", "have a nice day",
    "ok", "okay", "alright", "cool", "great", "nice", "noted",
    "got it", "sounds good", "perfect", "sure", "yep", "yup", "yeah",
    "start", "restart", "reset", "menu", "back", "hi again", "hello again",
}

def _greeting_reply(question: str) -> str | None:
    import random
    q = question.lower().strip().strip("!.,? ")
    words = q.split()
    # strip punctuation from words for matching
    clean = " ".join(w.strip("!.,?") for w in words)
    if clean not in _STREAM_GREETINGS and q not in _STREAM_GREETINGS:
        return None
    if any(w in q for w in ["bye", "goodbye", "see you", "later", "cya", "take care"]):
        return random.choice([
            "Take care! Come back anytime you have an insurance question. 😊",
            "Bye! I'm here whenever you need help with anything insurance-related.",
        ])
    if any(w in q for w in ["thank", "thanks", "ty", "thx", "cheers", "appreciate"]):
        return random.choice([
            "Happy to help! Anything else you want to know? 😊",
            "Anytime! That's what I'm here for. Any other questions?",
            "Glad I could help! Ask away if you need anything else.",
        ])
    if any(w in q for w in ["morning", "afternoon", "evening", "night", "gm", "gn"]):
        return random.choice([
            "Hey, good to see you! Got any insurance questions I can help with?",
            "Hi! Hope your day's going well. What can I help you with today?",
        ])
    if any(w in q for w in ["how are", "how's it", "you doing", "you good", "how do you"]):
        return random.choice([
            "Doing great, thanks for asking! What insurance question can I help you with? 😊",
            "All good here! What can I help you with today?",
        ])
    if any(w in q for w in ["start", "restart", "reset", "menu", "back"]):
        return "Sure! Ask me anything about insurance — policies, coverage, claims, I've got you. 😊"
    return random.choice([
        "Hey! 👋 What insurance question can I help you with today?",
        "Hi there! Got an insurance question? I'm all ears. 😊",
        "Hello! What can I help you with today?",
        "Hey! Good to see you. What's on your mind?",
    ])


@app.post("/ask-stream")
async def ask_stream(req: AskRequest):
    """True token-streaming endpoint — yields LLM tokens as SSE text/plain.

    The final line is a JSON object: {"sources": [...], "done": true}
    which the frontend uses to display citations after the answer streams in.
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Block AI while a human agent is handling this session
    if req.session_id and req.session_id != "default":
        _hub_sess = _agent_hub.get_session(req.session_id)
        if _hub_sess and _hub_sess.status == "human":
            _agent_name = ""
            if _hub_sess.agent_id:
                _ag = next((a for a in _agent_hub._agents.values() if a.agent_id == _hub_sess.agent_id), None)
                if _ag:
                    _agent_name = f" with {_ag.name}"
            _block_msg = f"You're connected{_agent_name} right now — please use the chat above to message the agent directly."
            # Forward user's message to the agent so they can see it
            _q = req.question
            _sid = req.session_id
            async def _blocked_gen():
                asyncio.create_task(_agent_hub.log_message(_sid, "user", _q))
                yield _block_msg
                yield "\n\n" + _json.dumps({"sources": [], "done": True})
            return StreamingResponse(
                _blocked_gen(),
                media_type="text/plain",
                headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache, no-transform"},
            )

    # Fast-path: greetings bypass the LLM entirely — instant reply
    # Fast-path: ILLEGAL content — firm refusal, no RAG, no handoff
    _ILLEGAL_PATTERNS = re.compile(
        r"\b(bomb|explosive|kill|murder|suicide|weapon|poison|terror|"
        r"rape|porn|naked|drugs|cocaine|heroin|meth|steal|robbery|"
        r"fraud|scam|forge|launder|hack|malware|ransomware)\b"
    )
    if _ILLEGAL_PATTERNS.search(req.question.lower()):
        async def _illegal_gen():
            yield "I'm only here to help with insurance questions — please keep our conversation focused on insurance. 😊"
            yield "\n\n" + _json.dumps({"sources": [], "done": True, "needs_human": False})
        return StreamingResponse(
            _illegal_gen(),
            media_type="text/plain",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache, no-transform"},
        )

    # Fast-path: OFF_TOPIC — casual refusal, no RAG, no handoff
    _OFF_TOPIC_PATTERNS = re.compile(
        r"\b(what is the date|today.s date|current date|what day is it|"
        r"what time is it|current time|weather today|temperature today|"
        r"who is the president|who won the|capital of|population of|"
        r"how to cook|recipe for|"
        r"chatgpt|gpt-4|openai|gemini)\b"
    )
    if _OFF_TOPIC_PATTERNS.search(req.question.lower()):
        async def _offtopic_gen():
            yield "That's a bit outside what I do! I'm Layla, your insurance advisor — happy to help with anything insurance-related. 😊"
            yield "\n\n" + _json.dumps({"sources": [], "done": True, "needs_human": False})
        return StreamingResponse(
            _offtopic_gen(),
            media_type="text/plain",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache, no-transform"},
        )
    greeting_reply = _greeting_reply(req.question)
    if greeting_reply:
        _sid = req.session_id
        async def _greeting_gen():
            yield greeting_reply
            yield "\n\n" + _json.dumps({"sources": [], "done": True})
            # Log to hub so agents can see greeting exchanges too
            if _sid and _sid != "default":
                _agent_hub.get_or_create_session(_sid)
                asyncio.create_task(_agent_hub.log_message(_sid, "user", req.question))
                asyncio.create_task(_agent_hub.log_message(_sid, "ai", greeting_reply))
        return StreamingResponse(
            _greeting_gen(),
            media_type="text/plain",
            headers={
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache, no-transform",
                "X-Content-Type-Options": "nosniff",
            },
        )

    async def generate():
        from multi_source_rag import _strip_markdown
        multi = _get_multi_rag()

        # Load conversation history so ask_stream can use it for context-aware
        # retrieval and follow-up question reformulation.
        history_list = await _get_conversation_history(req.session_id)
        history_str = "\n".join(
            f"{t['role'].capitalize()}: {t['content']}"
            for t in history_list[-_MAX_HISTORY_TURNS * 2:]
        )

        full_text = ""
        try:
            buf = ""
            async for token in multi.ask_stream(req.question, history=history_str, document_filter=None):
                if token.startswith('\n\n{'):
                    if buf:
                        stripped = _strip_markdown(buf)
                        full_text += stripped
                        yield stripped
                        await asyncio.sleep(0)
                    try:
                        final_data = _json.loads(token.strip())
                    except Exception:
                        final_data = {"sources": [], "done": True}
                    sources = final_data.get("sources", [])
                    # Prefer corrected_text over the raw streamed full_text: when
                    # multi_source_rag.py strips a Rule4 disclaimer the model
                    # appended after an already-correct, trusted answer, full_text
                    # (accumulated from tokens streamed BEFORE that correction) still
                    # contains the disclaimer — checking it here would independently
                    # re-trigger the human-handoff/offline-escalation flow even
                    # though the actual displayed answer no longer has the
                    # disclaimer at all.
                    _text_for_handoff_check = final_data.get("corrected_text") or full_text
                    # Read multi_source_rag.py's own grounding-check verdict before
                    # it gets overwritten below — only present (True) on refusal
                    # paths (reranker gate, coverage-check failure, empty context,
                    # Rule4 discard); absent on a normal successful generation, in
                    # which case there's no upstream complaint to add here.
                    _upstream_needs_human = final_data.get("needs_human", False)
                    ai_cant_answer = _agent_hub.response_needs_human(
                        _text_for_handoff_check, sources, _upstream_needs_human
                    )
                    agents_online  = _agent_hub.online_count() > 0
                    needs_human         = ai_cant_answer and agents_online
                    offline_escalated   = ai_cant_answer and not agents_online
                    final_data["needs_human"]       = needs_human
                    final_data["offline_escalated"] = offline_escalated
                    # Log to hub — auto-create session if missing (handles backend restarts)
                    if req.session_id and req.session_id != "default":
                        _agent_hub.get_or_create_session(req.session_id)
                        if offline_escalated:
                            # Must await so history is fully written before PDF is generated
                            await _agent_hub.log_message(req.session_id, "user", req.question)
                            await _agent_hub.log_message(req.session_id, "ai", _text_for_handoff_check)
                            asyncio.create_task(
                                _agent_hub.trigger_offline_escalation(req.session_id, req.question)
                            )
                        else:
                            asyncio.create_task(_agent_hub.log_message(req.session_id, "user", req.question))
                            asyncio.create_task(_agent_hub.log_message(req.session_id, "ai", _text_for_handoff_check))
                            if needs_human and agents_online:
                                # Await (not create_task) so the popup is sent to the agent
                                # BEFORE the final JSON reaches the frontend. This guarantees
                                # the question text is included and no race with frontend WS.
                                await _agent_hub.request_handoff(req.session_id, req.question)
                    # Persist this turn so the next question sees it as history — use the
                    # corrected text so a stripped Rule4 disclaimer doesn't leak into
                    # future-turn context and confuse follow-up reformulation.
                    if req.session_id and req.session_id != "default" and _text_for_handoff_check:
                        new_hist = history_list + [
                            {"role": "user", "content": req.question},
                            {"role": "assistant", "content": _text_for_handoff_check},
                        ]
                        # Must await (not create_task): the next request on this
                        # session reads history at the very start of generate()
                        # via _get_conversation_history(). A fire-and-forget save
                        # here raced against a fast-enough follow-up — the read
                        # could land before this write finished, silently making
                        # a real follow-up look like a fresh, history-less
                        # question. Reproduced with zero-delay back-to-back
                        # requests: a follow-up asking for "a real world example
                        # of that" answered about a completely unrelated topic
                        # because history was still empty when it was reformulated.
                        # The actual text has already been streamed to the client
                        # by this point, so this only delays the trailing metadata
                        # JSON chunk, not the perceived answer.
                        await _save_conversation_history(req.session_id, new_hist)
                    yield "\n\n" + _json.dumps(final_data)
                    return
                buf += token
                if ' ' in buf or '\n' in buf:
                    stripped = _strip_markdown(buf)
                    full_text += stripped
                    yield stripped
                    buf = ""
                    await asyncio.sleep(0)
            if buf:
                stripped = _strip_markdown(buf)
                full_text += stripped
                yield stripped
        except asyncio.TimeoutError:
            yield "\n\nError: The AI model server is taking too long. Please try again."
        except Exception as exc:
            logger.exception("Streaming ask failed")
            yield "\n\nError: Could not generate an answer due to an internal error."

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "X-Accel-Buffering": "no",          # nginx: disable proxy buffering
            "X-Vercel-Skip-Buffering": "1",      # Vercel Edge: stream chunks immediately
            "Cache-Control": "no-cache, no-store, no-transform",
            "Content-Encoding": "identity",      # disable gzip so Vercel can't buffer to compress
            "X-Content-Type-Options": "nosniff",
        },
    )


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
    # Exclude YouTube transcripts — they belong in the /videos section
    documents = [
        {"filename": s, "chunks": counts.get(s, 0)}
        for s in filenames
        if not s.startswith("youtube_")
    ]
    return {"documents": documents}


@app.delete("/docs")
async def clear_docs(_: str = Depends(require_auth)):
    await asyncio.to_thread(_get_pipeline().clear_documents)
    return {"status": "cleared"}


@app.post("/admin/cache/clear", tags=["admin"])
async def clear_kv_cache(_: str = Depends(require_auth)):
    """Clear the KV answer cache. All future queries will regenerate fresh answers."""
    multi = _get_multi_rag()
    cache = getattr(multi, "doc_pipeline", None)
    cache = getattr(cache, "_cache", None) if cache is not None else None
    if cache is None:
        raise HTTPException(status_code=503, detail="Cache not available.")
    await asyncio.to_thread(cache.clear)
    return {"status": "cleared"}


@app.delete("/docs/{name:path}")
async def remove_doc(name: str, _: str = Depends(require_auth)):
    pipeline = _get_pipeline()
    filenames_before = set(pipeline.vector_store.list_values("filename"))
    if name not in filenames_before:
        raise HTTPException(status_code=404, detail=f"Document '{name}' not found.")
    await asyncio.to_thread(pipeline.remove_document, name)
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


# ── REST aliases for frontend (/videos POST, /webpages POST) ─────────────────
# Frontend uses REST-idiomatic POST /videos and POST /webpages; register the
# same handlers under both paths without duplicating logic.
app.add_api_route("/videos",   upload_video,   methods=["POST"], dependencies=[Depends(require_auth)])
app.add_api_route("/webpages", upload_webpage, methods=["POST"], dependencies=[Depends(require_auth)])


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
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_community.embeddings import HuggingFaceEmbeddings

            from router import get_insurance_llm

            ragas_llm = get_insurance_llm(temperature=0, max_tokens=512)
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