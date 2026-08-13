"""
Human-agent handoff hub — manages chat sessions and WebSocket connections
for live agent monitoring and real-time conversation takeover.

Sessions live in Redis (session:{id} keys, see redis_client.py) so history
survives backend restarts and agent logouts, with self._sessions kept as
an in-memory write-through cache — every read stays synchronous exactly
as before (get_session, list_sessions, the WS handoff state machine),
only the persistence call at the end of each mutation is now `await
self._save_session(sid)` (one Redis key) instead of the old
_save_sessions() (rewrite the ENTIRE sessions_data.json, every session,
on every single message — the actual thing being fixed here). On first
boot after this change (Redis genuinely empty), startup() migrates
whatever was in the old sessions_data.json once; every boot after that
is a no-op since Redis already has session keys.

Session deletion (delete_session, delete_inactive_sessions) is still the
authoritative expiry mechanism, not Redis TTL — a session's Redis key
also carries a generous backstop TTL (_BACKSTOP_TTL_SECONDS) purely so an
abandoned key can't outlive the JSON file it replaced even if the
reaper task in api.py were ever down for an extended period, but under
normal operation the reaper always fires first. Relying on Redis TTL as
the PRIMARY mechanism was considered and rejected: self._sessions has no
way to notice a key silently expiring in Redis between reads, so it
would either drift out of sync with what's actually persisted or need
its own reconciliation pass — no simpler than just keeping the reaper.
"""
import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Set

logger = logging.getLogger(__name__)

from fastapi import WebSocket

import redis_client

_HERE = os.path.dirname(os.path.abspath(__file__))
_SESSIONS_FILE = os.path.join(_HERE, "sessions_data.json")  # legacy — read only, one-time migration
_AGENT_ACTIVITY_FILE = os.path.join(_HERE, "agent_activity.json")

_SESSION_KEY_PREFIX = "session:"
# 3x the inactivity-purge threshold (api.py's SESSION_INACTIVITY_HOURS,
# default 24h) — a safety net only; delete_inactive_sessions() is the
# mechanism that's actually supposed to fire first. See module docstring.
_BACKSTOP_TTL_SECONDS = int(os.getenv("SESSION_INACTIVITY_HOURS", "24")) * 3 * 3600


def _session_key(session_id: str) -> str:
    return f"{_SESSION_KEY_PREFIX}{session_id}"


def _session_to_dict(session: "ChatSession") -> dict:
    return {
        "created_at": session.created_at,
        "tone": session.tone,
        "handoff_exhausted": session.handoff_exhausted,
        "email_sent": session.email_sent,
        "active_agent": session.active_agent,
        "awaiting_agent_confirmation": session.awaiting_agent_confirmation,
        "history": [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp, "meta": m.meta}
            for m in session.history
        ],
    }


def _session_from_dict(session_id: str, data: dict) -> "ChatSession":
    session = ChatSession(
        session_id=session_id,
        status="ai",  # always "ai" on load; no agents connected yet
        created_at=data.get("created_at", _now_full()),
        tone=data.get("tone", "neutral"),
        handoff_exhausted=data.get("handoff_exhausted", False),
        email_sent=data.get("email_sent", False),
        active_agent=data.get("active_agent", "layla"),
        awaiting_agent_confirmation=data.get("awaiting_agent_confirmation", ""),
    )
    for m in data.get("history", []):
        session.history.append(ChatMessage(
            role=m["role"],
            content=m["content"],
            timestamp=m.get("timestamp", ""),
            meta=m.get("meta", {}),
        ))
    return session


def _now() -> str:
    # ISO 8601 with an explicit UTC offset (not a bare "HH:MM") so the frontend
    # can actually convert it to the viewer's local time zone. A bare "HH:MM"
    # string already threw away the date and offset, so any attempt to display
    # it "in local time" was just echoing the server's UTC clock — every
    # non-UTC viewer saw a wrong time with no way to correct for it after the
    # fact. Every ChatMessage.timestamp goes through this function, so this is
    # the single choke point for the fix.
    return datetime.now(timezone.utc).isoformat()

def _now_full() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def _last_activity(session: "ChatSession") -> datetime:
    """Timestamp of the most recent thing that happened in *session* — the
    last message if there is one, else its creation time. Used by the
    inactivity reaper below; parse failures return "now" (never stale) so
    a malformed/legacy timestamp can never cause a session to be purged.
    Message timestamps are full ISO 8601 with a UTC offset (_now());
    created_at is the older "%Y-%m-%d %H:%M" UTC-naive format (_now_full())
    — both are handled here since sessions loaded from disk may predate
    either format change."""
    raw = session.history[-1].timestamp if session.history else session.created_at
    try:
        dt = datetime.fromisoformat(raw)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        pass
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


_ABUSE_WORDS = frozenset({
    "fuck", "fucking", "fucked", "shit", "shitty", "bitch", "bitches",
    "bastard", "bastards", "asshole", "assholes", "idiot", "idiots",
    "moron", "morons", "stupid", "damn", "crap", "piss", "dick",
    "jerk", "dumb", "scam", "fraud", "useless", "garbage", "trash",
    "pathetic", "disgusting", "horrible", "awful", "scammer", "liar",
    "cheat", "ridiculous", "bullshit", "nonsense", "incompetent",
    "worthless", "terrible", "worst", "clueless", "rubbish", "crook",
})

async def _analyze_tone(text: str) -> str:
    """Return 'happy', 'angry', or 'neutral'. Abuse words fast-path before LLM call."""
    words = set(re.findall(r'\b[a-z]+\b', text.lower()))
    if words & _ABUSE_WORDS:
        return "angry"
    if len(text.strip()) < 15:
        return "neutral"
    try:
        import aiohttp
        from router import VLLM_HOST, VLLM_API_KEY, _resolve_vllm_model
        if not VLLM_HOST:
            return "neutral"
        prompt = (
            "You are a tone classifier for customer service. Read the user message below and reply "
            "with exactly one word — nothing else.\n"
            "  happy   — user sounds satisfied, pleased, grateful, or positive\n"
            "  angry   — user sounds upset, frustrated, complaining, or demanding\n"
            "  neutral — anything else\n\n"
            f"Message: {text[:300]}\n"
            "Tone:"
        )
        payload = {
            "model": _resolve_vllm_model(),
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4,
            "temperature": 0,
            "stream": False,
        }
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession() as sess:
            async with sess.post(
                f"{VLLM_HOST}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {VLLM_API_KEY}"},
                timeout=timeout,
            ) as resp:
                if resp.status != 200:
                    return "neutral"
                data = await resp.json()
                result = data["choices"][0]["message"]["content"].strip().lower()
                if "happy" in result or "satisf" in result or "positive" in result:
                    return "happy"
                if "angry" in result or "upset" in result or "frustrat" in result:
                    return "angry"
                return "neutral"
    except Exception:
        return "neutral"


@dataclass
class ChatMessage:
    role: str   # "user" | "ai" | "agent" | "system"
    content: str
    timestamp: str = field(default_factory=_now)
    meta: dict = field(default_factory=dict)  # e.g. {"escalation_sent": True}


@dataclass
class ChatSession:
    session_id: str
    history: List[ChatMessage] = field(default_factory=list)
    status: str = "ai"          # "ai" | "waiting" | "human"
    agent_id: Optional[str] = None
    user_ws: Optional[WebSocket] = None
    created_at: str = field(default_factory=_now_full)
    tone: str = "neutral"       # "happy" | "neutral" | "angry"
    tone_from_red: bool = False  # True if agent took over because tone was "angry"
    handoff_exhausted: bool = False  # True after handoff timed-out/all-declined; resets on next AI turn
    email_sent: bool = False    # True after escalation email sent; cleared when agent takes over
    pending_ws_message: Optional[dict] = None  # buffered for when WS reconnects after timeout
    active_agent: str = "layla"          # "layla" | "ava" | future: "health", "life", etc.
    awaiting_agent_confirmation: str = "" # set to a target agent name when Layla has asked
                                          # "want to connect with X?" but the user hasn't
                                          # confirmed yet; empty string = no pending offer


@dataclass
class HumanAgent:
    agent_id: str
    name: str
    ws: WebSocket
    active_session: Optional[str] = None
    monitoring: Set[str] = field(default_factory=set)
    declined_sessions: Set[str] = field(default_factory=set)
    login_time: str = field(default_factory=_now_full)
    blocked: bool = False


_HANDOFF_TIMEOUT = 30  # seconds agents have to accept before email is sent


class AgentHub:

    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._agents: Dict[str, HumanAgent] = {}
        self._pending_handoffs: Dict[str, asyncio.Task] = {}
        self._agent_records: Dict[str, dict] = {}   # name → persistent activity record
        self._super_admin_ws: List[WebSocket] = []  # connected super-admin sockets
        self._super_admin_tokens: set = set()       # valid super-admin session tokens
        # Per-session write lock (see _save_session) — several call sites
        # fire off two log_message() calls back-to-back via
        # asyncio.create_task (e.g. "user" then "ai") without awaiting
        # either. That was harmless under the old sync file write (no
        # await point inside it, so whichever task started first always
        # finished first, in order). A real Redis round-trip DOES have a
        # suspension point, so without this lock the two SETs to the same
        # key can complete out of order — confirmed live: the "user"-only
        # write occasionally landed AFTER the "ai" write and silently
        # erased the AI reply from the session's persisted history. The
        # lock only serializes the SET itself; _session_to_dict() still
        # reads self._sessions[sid].history fresh at write time, so the
        # second writer under the lock always captures whatever the first
        # writer already appended.
        self._session_locks: Dict[str, asyncio.Lock] = {}
        self._load_agent_records()
        # self._sessions is populated by startup() (async — Redis needs an
        # await), called from api.py's FastAPI startup hook. Left empty
        # here rather than blocking __init__, since `hub = AgentHub()`
        # below runs at import time, before any event loop exists to await
        # on.

    # ── Persistence ───────────────────────────────────────────────────────────

    async def startup(self) -> None:
        """Load every session from Redis into self._sessions. One-time
        migration: if Redis has zero session keys (first boot after this
        store moved off sessions_data.json) AND that legacy file still
        exists with data, import it into Redis and memory instead of
        starting empty — every boot after this one finds session keys
        already in Redis and this branch never runs again."""
        keys = await redis_client.scan_keys(f"{_SESSION_KEY_PREFIX}*")
        if not keys:
            migrated = self._load_sessions_from_legacy_file()
            if migrated:
                for sid, session in migrated.items():
                    self._sessions[sid] = session
                    await redis_client.set_json(
                        _session_key(sid), _session_to_dict(session), _BACKSTOP_TTL_SECONDS
                    )
                logger.info("Migrated %d session(s) from legacy sessions_data.json into Redis.", len(migrated))
            return
        for key in keys:
            data = await redis_client.get_json(key)
            if not data:
                continue
            sid = key[len(_SESSION_KEY_PREFIX):]
            self._sessions[sid] = _session_from_dict(sid, data)
        logger.info("Loaded %d session(s) from Redis.", len(self._sessions))

    @staticmethod
    def _load_sessions_from_legacy_file() -> Dict[str, "ChatSession"]:
        if not os.path.exists(_SESSIONS_FILE):
            return {}
        try:
            with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {sid: _session_from_dict(sid, s) for sid, s in data.items()}
        except Exception:
            logger.exception("Could not read legacy sessions_data.json for migration")
            return {}

    async def _save_session(self, session_id: str) -> None:
        """Persist ONE session to Redis — replaces the old _save_sessions(),
        which rewrote every session in the store on every single message.
        No-op if the session isn't in memory (e.g. already deleted).
        Serialized per-session (see self._session_locks) so two writes to
        the same key from concurrent fire-and-forget log_message() calls
        can't complete out of order and silently drop one of them."""
        if session_id not in self._sessions:
            return
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            await redis_client.set_json(_session_key(session_id), _session_to_dict(session), _BACKSTOP_TTL_SECONDS)

    # ── Agent-activity persistence ────────────────────────────────────────────

    def _load_agent_records(self):
        if not os.path.exists(_AGENT_ACTIVITY_FILE):
            return
        try:
            with open(_AGENT_ACTIVITY_FILE, "r", encoding="utf-8") as f:
                self._agent_records = json.load(f)
        except Exception:
            self._agent_records = {}

    def _save_agent_records(self):
        try:
            tmp = _AGENT_ACTIVITY_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._agent_records, f, ensure_ascii=False, indent=2)
            os.replace(tmp, _AGENT_ACTIVITY_FILE)
        except Exception:
            pass

    def _ensure_agent_record(self, name: str) -> dict:
        if name not in self._agent_records:
            self._agent_records[name] = {
                "blocked": False,
                "login_sessions": [],
                "chats": [],
                "total_queries_answered": 0,
            }
        return self._agent_records[name]

    # ── Sessions ──────────────────────────────────────────────────────────────

    async def create_session(self) -> str:
        sid = uuid.uuid4().hex[:8]
        self._sessions[sid] = ChatSession(session_id=sid)
        await self._save_session(sid)
        return sid

    async def delete_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        if session.agent_id and session.agent_id in self._agents:
            self._agents[session.agent_id].active_session = None
        if session.user_ws:
            try:
                await session.user_ws.send_json({
                    "type": "session_deleted",
                    "message": "This conversation was removed by an administrator.",
                })
            except Exception:
                pass
        del self._sessions[session_id]
        self._session_locks.pop(session_id, None)
        await redis_client.delete_key(_session_key(session_id))
        # Also prune this session from every agent's persistent chat-history log
        # (_agent_records[name]["chats"]) — that log survives independently of
        # self._sessions and used to be untouched by delete, so a deleted
        # session's card stayed visible forever in the super-admin's per-agent
        # "Chats" tab. Clicking Delete on that lingering card always failed
        # with "not found" (the live session really was gone), which is where
        # the confusing "already deleted" error was coming from. Pruning here
        # makes the card disappear everywhere delete is expected to reach.
        _records_changed = False
        for rec in self._agent_records.values():
            _chats = rec.get("chats")
            if not _chats:
                continue
            _kept = [c for c in _chats if c.get("session_id") != session_id]
            if len(_kept) != len(_chats):
                rec["chats"] = _kept
                _records_changed = True
        if _records_changed:
            self._save_agent_records()
        # Notify every connected agent to remove this session from their view immediately
        for agent in list(self._agents.values()):
            agent.monitoring.discard(session_id)
            try:
                await agent.ws.send_json({
                    "type": "session_removed",
                    "session_id": session_id,
                })
            except Exception:
                pass
        await self._broadcast_sessions_update()
        return True

    async def delete_inactive_sessions(self, hours: float = 24) -> List[str]:
        """Purge every session whose last activity is older than *hours*
        (storage-growth control — Redis's own backstop TTL is a safety net
        only, this is the mechanism actually meant to fire first; see the
        module docstring). Skips a session with a live
        user_ws even if its last logged message is stale, so an open tab
        idling past the cutoff is never pulled out from under a connected
        user. Reuses delete_session() for each one, so a purged session
        gets the exact same cleanup (agent-record pruning, connected-agent
        notification, super-admin broadcast) a manual admin delete gets.
        Returns the ids actually deleted, so callers can purge the same
        ids from every OTHER session_id-keyed store (conversations.json,
        conversation_agent_sessions.json, travel_bot's DB) — this method
        only owns its own store."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        stale_ids = [
            sid for sid, s in list(self._sessions.items())
            if s.user_ws is None and _last_activity(s) < cutoff
        ]
        deleted = []
        for sid in stale_ids:
            if await self.delete_session(sid):
                deleted.append(sid)
        return deleted

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    async def get_or_create_session(self, session_id: str) -> "ChatSession":
        """Return existing session or create one on the fly (handles backend restarts)."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(session_id=session_id)
            await self._save_session(session_id)
        return self._sessions[session_id]

    def list_sessions(self) -> List[dict]:
        out = []
        for s in sorted(self._sessions.values(), key=lambda x: x.created_at, reverse=True):
            last = s.history[-1].content[:80] if s.history else ""
            first_user = next((m.content[:60] for m in s.history if m.role == "user"), None)
            # For live sessions the status is authoritative ("human", "waiting", "ai").
            # For historical sessions loaded from disk (always "ai"), derive a richer
            # display status from history so the sidebar shows meaningful colors:
            # sessions where a human agent ever responded show green (human).
            display_status = s.status
            # Only override to "human" if an agent is ACTIVELY assigned right now.
            # Without this guard, released sessions (agent_id=None, status="ai") were
            # incorrectly shown as "human" because history contained agent messages.
            if display_status == "ai" and s.agent_id and any(m.role == "agent" for m in s.history):
                display_status = "human"
            out.append({
                "session_id": s.session_id,
                "status": display_status,
                "agent_id": s.agent_id,
                "message_count": len(s.history),
                "created_at": s.created_at,
                "last_message": last,
                "title": first_user or f"Session #{s.session_id}",
                "tone": s.tone,
                "tone_from_red": s.tone_from_red,
                "email_sent": getattr(s, "email_sent", False),
                "active_agent": s.active_agent,
                "awaiting_agent_confirmation": s.awaiting_agent_confirmation,
            })
        return out

    async def log_message(self, session_id: str, role: str, content: str):
        session = self._sessions.get(session_id)
        if not session:
            return
        msg = ChatMessage(role=role, content=content)
        session.history.append(msg)
        if role == "ai":
            session.handoff_exhausted = False  # fresh AI turn — allow handoff again if needed
        await self._save_session(session_id)
        await self._broadcast_new_message(session_id, msg)
        await self._broadcast_sessions_update()
        if role == "user":
            asyncio.create_task(self._analyze_and_broadcast_tone(session_id, content))

    # ── Agent registration ────────────────────────────────────────────────────

    def register_agent(self, agent_id: str, name: str, ws: WebSocket) -> "HumanAgent":
        rec = self._ensure_agent_record(name)
        agent = HumanAgent(
            agent_id=agent_id, name=name, ws=ws,
            login_time=_now_full(), blocked=rec.get("blocked", False),
        )
        self._agents[agent_id] = agent
        rec["login_sessions"].append({
            "agent_id": agent_id,
            "login_time": agent.login_time,
            "logout_time": None,
            "duration_minutes": None,
        })
        self._save_agent_records()
        return agent

    async def unregister_agent(self, agent_id: str):
        agent = self._agents.pop(agent_id, None)
        if not agent:
            return
        if agent.active_session:
            session = self._sessions.get(agent.active_session)
            if session:
                session.status = "ai"
                session.agent_id = None
                session.tone_from_red = False
                await self._save_session(agent.active_session)
                if session.user_ws:
                    try:
                        await session.user_ws.send_json({
                            "type": "agent_left",
                            "message": "The agent disconnected. You're back with Layla.",
                        })
                    except Exception:
                        pass
        # Record logout time + duration
        rec = self._agent_records.get(agent.name)
        if rec:
            now_str = _now_full()
            fmt = "%Y-%m-%d %H:%M"
            for sess in reversed(rec["login_sessions"]):
                if sess.get("agent_id") == agent_id and sess.get("logout_time") is None:
                    sess["logout_time"] = now_str
                    try:
                        login_dt = datetime.strptime(sess["login_time"], fmt).replace(tzinfo=timezone.utc)
                        logout_dt = datetime.strptime(now_str, fmt).replace(tzinfo=timezone.utc)
                        sess["duration_minutes"] = round((logout_dt - login_dt).total_seconds() / 60, 1)
                    except Exception:
                        pass
                    break
            # Close any open chat record for this agent
            for chat in rec["chats"]:
                if chat.get("agent_id") == agent_id and chat.get("ended_at") is None:
                    chat["ended_at"] = now_str
            self._save_agent_records()
        await self._broadcast_sessions_update()
        await self._broadcast_super_admin_update()

    def online_count(self) -> int:
        # Only count agents whose WebSocket connection is still open
        from starlette.websockets import WebSocketState
        return sum(
            1 for a in self._agents.values()
            if a.ws.client_state == WebSocketState.CONNECTED
        )

    # ── Handoff ───────────────────────────────────────────────────────────────

    async def request_handoff(self, session_id: str, question: str = "") -> bool:
        """
        New flow: broadcast a popup to all free agents instead of auto-assigning.
        Returns True if at least one agent was notified, False if no agents online.
        If no agents are online the caller is responsible for offline escalation.
        Pass `question` directly to avoid a race condition where session history
        isn't written yet when this is called as a background task.
        """
        session = self._sessions.get(session_id)
        if not session:
            return False
        if session.status == "human":
            return True  # already has an agent
        if session_id in self._pending_handoffs:
            return True  # popup already sent, still waiting
        if session.handoff_exhausted:
            return False  # already timed-out or all-declined this turn — don't re-popup

        # Notify all connected agents who haven't declined this session.
        # Busy agents (with active_session) are still included — accept_handoff
        # already handles releasing their current session on accept.
        notifiable = [
            a for a in self._agents.values()
            if session_id not in a.declined_sessions
        ]
        logger.info("request_handoff: session=%s agents_total=%d notifiable=%d",
                    session_id, len(self._agents), len(notifiable))
        if not notifiable:
            logger.warning("request_handoff: no agents online — falling back to email")
            return False  # caller sends email

        session.status = "waiting"
        await self._save_session(session_id)

        # Prefer the caller-supplied question; fall back to last user message in history
        unanswerable = question or next(
            (m.content for m in reversed(session.history) if m.role == "user"), ""
        )
        title = session.history[0].content[:60] if session.history else f"Session #{session_id}"

        # Send popup to every available agent (including those handling another session)
        popup_msg = {
            "type": "handoff_request",
            "session_id": session_id,
            "title": title,
            "query": unanswerable,
            "message_count": len(session.history),
            "timeout": _HANDOFF_TIMEOUT,
        }
        for agent in notifiable:
            try:
                await agent.ws.send_json(popup_msg)
            except Exception:
                pass

        # Start timeout — if nobody accepts, send email and release
        task = asyncio.create_task(self._handoff_timeout(session_id, unanswerable))
        self._pending_handoffs[session_id] = task

        await self._broadcast_sessions_update()
        return True

    async def cancel_handoff(self, session_id: str) -> bool:
        """User explicitly cancelled waiting for a human agent — release back
        to AI and email the team, same outcome as a natural handoff timeout.
        Shared by both the WebSocket handler and the HTTP fallback below (the
        WS-only version had no fallback, so a dropped WebSocket meant Cancel
        silently did nothing server-side — the client optimistically flipped
        its own UI back to "ai", but the very next poll saw the session was
        still "waiting" and flipped it right back, making the button look
        completely unresponsive).

        Returns True if there was an actual pending handoff cancelled, False
        if the session wasn't in "waiting" state (nothing to do).
        """
        session = self._sessions.get(session_id)
        if not session or session.status != "waiting":
            return False
        task = self._pending_handoffs.pop(session_id, None)
        if task:
            task.cancel()
        session.status = "ai"
        session.handoff_exhausted = True
        session.email_sent = True
        await self._save_session(session_id)
        unanswerable = next(
            (m.content for m in reversed(session.history) if m.role == "user"), ""
        )
        asyncio.create_task(self.trigger_offline_escalation(session_id, unanswerable))
        await self._broadcast_sessions_update()
        return True

    async def _handoff_timeout(self, session_id: str, unanswerable_query: str):
        """Called after _HANDOFF_TIMEOUT seconds if no agent accepted the popup."""
        await asyncio.sleep(_HANDOFF_TIMEOUT)
        if session_id not in self._pending_handoffs:
            return  # already accepted — task was cancelled
        self._pending_handoffs.pop(session_id, None)
        session = self._sessions.get(session_id)
        if not session or session.status != "waiting":
            return

        session.status = "ai"
        session.handoff_exhausted = True
        session.email_sent = True
        # Tag the triggering user message so the super-admin "Agent Only" view can flag it,
        # and so it surfaces in the agent dashboard's per-session "Unanswered Queries" panel
        escalated_index = None
        for i in range(len(session.history) - 1, -1, -1):
            if session.history[i].role == "user":
                session.history[i].meta["escalation_sent"] = True
                escalated_index = i
                break
        _timeout_msg = {
            "type": "handoff_timeout",
            "message": "No agent responded in time. We've emailed our support team, someone will reach out to you soon.",
        }
        delivered = False
        if session.user_ws:
            try:
                await session.user_ws.send_json(_timeout_msg)
                delivered = True
            except Exception:
                pass
        if not delivered:
            session.pending_ws_message = _timeout_msg
        await self._save_session(session_id)
        if escalated_index is not None:
            await self._broadcast_message_meta_update(session_id, escalated_index, session.history[escalated_index].meta)
        import asyncio as _aio
        history_snapshot = list(session.history)
        await _aio.to_thread(_send_email_sync, session_id, history_snapshot, unanswerable_query)
        await self._broadcast_sessions_update()

    async def accept_handoff(self, agent_id: str, session_id: str):
        """Agent accepted the popup — assign and cancel the timeout."""
        session = self._sessions.get(session_id)
        agent   = self._agents.get(agent_id)
        if not session or not agent or agent.blocked:
            return

        # Cancel the timeout task if still running (may be missing after server restart)
        task = self._pending_handoffs.pop(session_id, None)
        if task:
            task.cancel()

        # Guard: only assign if the session is still waiting (prevents double-accept)
        if session.status not in ("waiting", "ai"):
            # Already taken by someone else
            try:
                await agent.ws.send_json({"type": "handoff_fulfilled", "session_id": session_id})
            except Exception:
                pass
            return

        await self._assign_agent(session, agent)

        # Tell all other agents the request was fulfilled and clear their declined record
        for a in self._agents.values():
            a.declined_sessions.discard(session_id)
            if a.agent_id != agent_id:
                try:
                    await a.ws.send_json({"type": "handoff_fulfilled", "session_id": session_id})
                except Exception:
                    pass

    async def decline_handoff(self, agent_id: str, session_id: str):
        """Agent dismissed the popup. If no other free agents remain, send email immediately."""
        agent = self._agents.get(agent_id)
        if agent:
            agent.declined_sessions.add(session_id)

        # Check whether any connected agent (including busy ones) can still accept
        can_still_accept = [
            a for a in self._agents.values()
            if session_id not in a.declined_sessions
        ]
        if can_still_accept:
            return  # others may still accept — let the timer keep running

        # Nobody left — cancel the timeout and escalate immediately
        task = self._pending_handoffs.pop(session_id, None)
        if task:
            task.cancel()

        session = self._sessions.get(session_id)
        if not session or session.status != "waiting":
            return

        session.status = "ai"
        session.agent_id = None
        session.handoff_exhausted = True
        session.email_sent = True
        # Tag the triggering user message so the super-admin "Agent Only" view can flag it,
        # and so it surfaces in the agent dashboard's per-session "Unanswered Queries" panel
        escalated_index = None
        for i in range(len(session.history) - 1, -1, -1):
            if session.history[i].role == "user":
                session.history[i].meta["escalation_sent"] = True
                escalated_index = i
                break
        _decline_msg = {
            "type": "handoff_timeout",
            "message": "No agent is available right now. We've notified our support team and someone will follow up with you shortly.",
        }
        delivered = False
        if session.user_ws:
            try:
                await session.user_ws.send_json(_decline_msg)
                delivered = True
            except Exception:
                pass
        if not delivered:
            session.pending_ws_message = _decline_msg
        await self._save_session(session_id)
        if escalated_index is not None:
            await self._broadcast_message_meta_update(session_id, escalated_index, session.history[escalated_index].meta)

        unanswerable = next(
            (m.content for m in reversed(session.history) if m.role == "user"), ""
        )
        history_snapshot = list(session.history)
        import asyncio as _aio
        await _aio.to_thread(_send_email_sync, session_id, history_snapshot, unanswerable)
        await self._broadcast_sessions_update()

    async def trigger_offline_escalation(self, session_id: str, unanswerable_query: str):
        """Called directly when NO agents are online at the time the AI can't answer."""
        import asyncio as _aio
        session = self._sessions.get(session_id)
        escalated_index = None
        if session:
            session.email_sent = True
            # Tag the triggering user message so the super-admin "Agent Only" view can flag it,
            # and so it surfaces in the agent dashboard's per-session "Unanswered Queries" panel
            for i in range(len(session.history) - 1, -1, -1):
                if session.history[i].role == "user":
                    session.history[i].meta["escalation_sent"] = True
                    escalated_index = i
                    break
            await self._save_session(session_id)
        history_snapshot = list(session.history) if session else []
        await _aio.to_thread(_send_email_sync, session_id, history_snapshot, unanswerable_query)
        if session and escalated_index is not None:
            await self._broadcast_message_meta_update(session_id, escalated_index, session.history[escalated_index].meta)
        await self._broadcast_sessions_update()

    async def _assign_agent(self, session: "ChatSession", agent: "HumanAgent"):
        # ── Release the agent's existing session first (if any) ────────────────
        if agent.active_session and agent.active_session != session.session_id:
            old_session = self._sessions.get(agent.active_session)
            if old_session and old_session.agent_id == agent.agent_id:
                old_session.status = "ai"
                old_session.agent_id = None
                await self._save_session(agent.active_session)
                if old_session.user_ws:
                    try:
                        await old_session.user_ws.send_json({
                            "type": "agent_left",
                            "message": "The agent is now assisting someone else. Layla is back to help!",
                        })
                    except Exception:
                        pass

        session.status = "human"
        session.agent_id = agent.agent_id
        session.email_sent = False
        if session.tone == "angry":
            session.tone_from_red = True
        agent.active_session = session.session_id
        agent.monitoring.add(session.session_id)
        await self._save_session(session.session_id)
        # Open or reopen a chat record in agent activity.
        # Deduplicate: if a record already exists for this session_id, reopen it
        # instead of creating a new one — prevents duplicate entries on re-login.
        rec = self._ensure_agent_record(agent.name)
        snapshot_msgs = [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in session.history
        ]
        existing_chat = next(
            (c for c in rec["chats"] if c.get("session_id") == session.session_id),
            None,
        )
        if existing_chat:
            existing_chat["ended_at"] = None
            existing_chat["agent_id"] = agent.agent_id
            existing_chat["messages"] = snapshot_msgs  # refresh snapshot
        else:
            rec["chats"].append({
                "session_id": session.session_id,
                "agent_id": agent.agent_id,
                "started_at": _now_full(),
                "ended_at": None,
                "messages": snapshot_msgs,
                "reply_count": 0,
            })
        self._save_agent_records()
        history_payload = [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp, "meta": m.meta, "index": idx}
            for idx, m in enumerate(session.history[-100:], start=max(0, len(session.history) - 100))
        ]
        try:
            await agent.ws.send_json({
                "type": "assigned",
                "session_id": session.session_id,
                "history": history_payload,
            })
        except Exception:
            pass
        if session.user_ws:
            try:
                await session.user_ws.send_json({
                    "type": "agent_joined",
                    "agent_name": agent.name,
                })
            except Exception:
                pass
        await self._broadcast_sessions_update()

    # ── Agent actions ─────────────────────────────────────────────────────────

    async def agent_monitor(self, agent_id: str, session_id: str):
        agent = self._agents.get(agent_id)
        session = self._sessions.get(session_id)
        if not agent or not session:
            return
        agent.monitoring.add(session_id)
        # Cap at last 100 messages so the WebSocket payload stays manageable
        history_payload = [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp, "meta": m.meta, "index": idx}
            for idx, m in enumerate(session.history[-100:], start=max(0, len(session.history) - 100))
        ]
        try:
            await agent.ws.send_json({
                "type": "history",
                "session_id": session_id,
                "history": history_payload,
            })
        except Exception:
            pass

    async def agent_takeover(self, agent_id: str, session_id: str):
        agent = self._agents.get(agent_id)
        session = self._sessions.get(session_id)
        if not agent or not session or agent.blocked:
            return
        # Exclusive lock: block takeover if another agent already owns this session
        if session.status == "human" and session.agent_id and session.agent_id != agent_id:
            try:
                other = self._agents.get(session.agent_id)
                other_name = other.name if other else "another agent"
                await agent.ws.send_json({
                    "type": "error",
                    "message": f"Session #{session_id} is locked by {other_name}. You can only take over after they hand back to AI.",
                })
            except Exception:
                pass
            return
        if agent.active_session and agent.active_session != session_id:
            old = self._sessions.get(agent.active_session)
            if old:
                old.status = "ai"
                old.agent_id = None
                await self._save_session(agent.active_session)
                if old.user_ws:
                    try:
                        await old.user_ws.send_json({
                            "type": "agent_left",
                            "message": "The agent has stepped away. Layla is back to help!",
                        })
                    except Exception:
                        pass
        await self._assign_agent(session, agent)

    async def agent_release(self, agent_id: str):
        agent = self._agents.get(agent_id)
        if not agent or not agent.active_session:
            return
        released_sid = agent.active_session
        session = self._sessions.get(released_sid)
        if session:
            session.status = "ai"
            session.agent_id = None
            session.tone = "neutral"
            session.tone_from_red = False
            await self._save_session(released_sid)
            if session.user_ws:
                try:
                    await session.user_ws.send_json({
                        "type": "agent_left",
                        "message": "The agent has stepped away. Layla is back to help!",
                    })
                except Exception:
                    pass
        # Close the chat record
        rec = self._agent_records.get(agent.name)
        if rec:
            for chat in reversed(rec["chats"]):
                if chat.get("session_id") == released_sid and chat.get("agent_id") == agent_id and chat.get("ended_at") is None:
                    chat["ended_at"] = _now_full()
                    break
            self._save_agent_records()
        agent.active_session = None
        await self._broadcast_sessions_update()
        await self._broadcast_super_admin_update()

    async def agent_send_message(self, agent_id: str, session_id: str, content: str):
        agent = self._agents.get(agent_id)
        session = self._sessions.get(session_id)
        if not agent or not session:
            return
        # Only the assigned agent may send messages to a locked session
        if session.agent_id and session.agent_id != agent_id:
            try:
                await agent.ws.send_json({
                    "type": "error",
                    "message": "You are not the assigned agent for this session.",
                })
            except Exception:
                pass
            return
        msg = ChatMessage(role="agent", content=content)
        session.history.append(msg)
        await self._save_session(session_id)
        # Record the reply in agent activity
        rec = self._agent_records.get(agent.name)
        if rec:
            for chat in reversed(rec["chats"]):
                if chat.get("session_id") == session_id and chat.get("agent_id") == agent_id and chat.get("ended_at") is None:
                    chat["reply_count"] = chat.get("reply_count", 0) + 1
                    chat["messages"].append({"role": "agent", "content": content, "timestamp": msg.timestamp})
                    rec["total_queries_answered"] = rec.get("total_queries_answered", 0) + 1
                    break
            self._save_agent_records()
        await self._broadcast_new_message(session_id, msg)
        await self._broadcast_sessions_update()
        await self._broadcast_super_admin_update()
        if session.user_ws:
            try:
                await session.user_ws.send_json({
                    "type": "agent_message",
                    "content": content,
                    "agent_name": agent.name,
                })
            except Exception:
                pass

    async def agent_answer_unanswered(self, agent_id: str, session_id: str, message_index: int, content: str):
        """
        Lightweight reply to ONE specific escalated ("unanswered") user query, sent
        from the per-session "Unanswered Queries" panel. Unlike agent_send_message,
        this does NOT assign/lock the session to this agent — Layla keeps answering
        any new questions normally, and any online agent (not just the one assigned,
        if any) can resolve these, since the whole point is answering without taking
        over ownership of the conversation.
        """
        agent = self._agents.get(agent_id)
        session = self._sessions.get(session_id)
        if not agent or not session:
            return
        if message_index < 0 or message_index >= len(session.history):
            return
        target = session.history[message_index]
        if target.role != "user" or not target.meta.get("escalation_sent") or target.meta.get("escalation_resolved"):
            return

        target.meta["escalation_resolved"] = True
        target.meta["resolved_by"] = agent.name

        reply = ChatMessage(role="agent", content=content, meta={"answers_index": message_index})
        session.history.append(reply)

        # Sidebar red dot stays on until every escalated query in this session is resolved
        session.email_sent = any(
            m.role == "user" and m.meta.get("escalation_sent") and not m.meta.get("escalation_resolved")
            for m in session.history
        )
        await self._save_session(session_id)

        rec = self._ensure_agent_record(agent.name)
        rec["total_queries_answered"] = rec.get("total_queries_answered", 0) + 1
        # This reply doesn't belong to any rec["chats"] entry — those track live
        # session takeovers (agent_send_message), and this path deliberately does
        # NOT assign/lock the session (see docstring above). get_super_admin_data's
        # "Today Replies" stat sums chats[].reply_count, so a reply sent from here
        # was previously invisible to it — total_queries_answered went up but the
        # per-day count never moved. Log it separately so both are covered without
        # forcing this into the chats structure's session-locking semantics.
        rec.setdefault("unanswered_reply_log", []).append(_now_full())
        self._save_agent_records()

        await self._broadcast_message_meta_update(session_id, message_index, target.meta)
        await self._broadcast_new_message(session_id, reply)
        await self._broadcast_sessions_update()
        await self._broadcast_super_admin_update()

        if session.user_ws:
            try:
                await session.user_ws.send_json({
                    "type": "agent_message",
                    "content": content,
                    "agent_name": agent.name,
                    "answers_question": target.content,
                })
            except Exception:
                pass

    async def user_message_to_agent(self, session_id: str, content: str):
        """Log a user message during human-agent mode and broadcast to monitoring agents."""
        session = self._sessions.get(session_id)
        if not session:
            return
        msg = ChatMessage(role="user", content=content)
        session.history.append(msg)
        await self._save_session(session_id)
        # Append user message to the active agent's chat record
        if session.agent_id:
            agent = self._agents.get(session.agent_id)
            if agent:
                rec = self._agent_records.get(agent.name)
                if rec:
                    for chat in reversed(rec["chats"]):
                        if chat.get("session_id") == session_id and chat.get("ended_at") is None:
                            chat["messages"].append({"role": "user", "content": content, "timestamp": msg.timestamp})
                            break
                    self._save_agent_records()
        await self._broadcast_new_message(session_id, msg)
        await self._broadcast_sessions_update()
        asyncio.create_task(self._analyze_and_broadcast_tone(session_id, content))

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _broadcast_new_message(self, session_id: str, msg: "ChatMessage"):
        session = self._sessions.get(session_id)
        index = (len(session.history) - 1) if session else None
        payload = {
            "type": "new_message",
            "session_id": session_id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
            "meta": msg.meta,
            "index": index,
        }
        for agent in list(self._agents.values()):
            if session_id in agent.monitoring or agent.active_session == session_id:
                try:
                    await agent.ws.send_json(payload)
                except Exception:
                    pass

    async def _broadcast_message_meta_update(self, session_id: str, index: int, meta: dict):
        """Tells any agent currently viewing this session that an existing message's
        meta flags changed (e.g. a query just got tagged/resolved as an escalation),
        so the 'Unanswered Queries' panel can update live without a history reload."""
        payload = {
            "type": "message_meta_updated",
            "session_id": session_id,
            "index": index,
            "meta": meta,
        }
        for agent in list(self._agents.values()):
            if session_id in agent.monitoring or agent.active_session == session_id:
                try:
                    await agent.ws.send_json(payload)
                except Exception:
                    pass

    async def _analyze_and_broadcast_tone(self, session_id: str, content: str):
        tone = await _analyze_tone(content)
        session = self._sessions.get(session_id)
        if not session:
            return
        session.tone = tone
        await self._broadcast_tone_update(session_id, tone, session.tone_from_red)
        await self._broadcast_sessions_update()

    async def _broadcast_tone_update(self, session_id: str, tone: str, from_red: bool = False):
        payload = {
            "type": "tone_update",
            "session_id": session_id,
            "tone": tone,
            "from_red": from_red,
        }
        for agent in list(self._agents.values()):
            if session_id in agent.monitoring or agent.active_session == session_id:
                try:
                    await agent.ws.send_json(payload)
                except Exception:
                    pass

    async def _broadcast_sessions_update(self):
        sessions = self.list_sessions()
        online_agents = [
            {
                "agent_id": a.agent_id,
                "name": a.name,
                "status": "chatting" if a.active_session else "online",
            }
            for a in self._agents.values()
        ]
        payload = {"type": "sessions_update", "sessions": sessions, "online_agents": online_agents}
        for agent in list(self._agents.values()):
            try:
                await agent.ws.send_json(payload)
            except Exception:
                pass

    # ── Super-admin support ───────────────────────────────────────────────────

    def register_super_admin(self, ws: WebSocket):
        self._super_admin_ws.append(ws)

    def unregister_super_admin(self, ws: WebSocket):
        self._super_admin_ws = [w for w in self._super_admin_ws if w is not ws]

    async def _broadcast_super_admin_update(self):
        if not self._super_admin_ws:
            return
        payload = {"type": "update", **self.get_super_admin_data()}
        for ws in list(self._super_admin_ws):
            try:
                await ws.send_json(payload)
            except Exception:
                pass

    def get_super_admin_data(self) -> dict:
        # Case-insensitive lookup — agent might log in as "Lavish" but record key is "lavish"
        online_names = {a.name.lower() for a in self._agents.values()}
        chatting_names = {a.name.lower() for a in self._agents.values() if a.active_session}
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        agents_out = []
        for name, rec in self._agent_records.items():
            name_lc = name.lower()
            if name_lc in chatting_names:
                status = "chatting"
            elif name_lc in online_names:
                status = "online"
            else:
                status = "offline"
            today_logins = [s for s in rec.get("login_sessions", []) if s.get("login_time", "").startswith(today)]
            today_hours = round(sum(s.get("duration_minutes") or 0 for s in today_logins) / 60, 2)
            today_replies = sum(c.get("reply_count", 0) for c in rec.get("chats", []) if c.get("started_at", "").startswith(today))
            # Replies sent from the "Unanswered Queries" panel (agent_answer_unanswered)
            # never create/update a chats[] entry — that path intentionally doesn't take
            # over the session — so they're logged separately and added here.
            today_replies += sum(1 for ts in rec.get("unanswered_reply_log", []) if ts.startswith(today))
            cur_login = next((s for s in reversed(rec.get("login_sessions", [])) if s.get("logout_time") is None), None)
            agents_out.append({
                "name": name,
                "status": status,
                "blocked": rec.get("blocked", False),
                "total_queries_answered": rec.get("total_queries_answered", 0),
                "today_queries": today_replies,
                "today_hours": today_hours,
                "current_login_time": cur_login.get("login_time") if cur_login else None,
                "login_sessions": rec.get("login_sessions", []),
            })
        # Add online agents not yet in records (first login race)
        known = {a["name"] for a in agents_out}
        for agent in self._agents.values():
            if agent.name not in known:
                agents_out.append({
                    "name": agent.name,
                    "status": "chatting" if agent.active_session else "online",
                    "blocked": agent.blocked,
                    "total_queries_answered": 0,
                    "today_queries": 0,
                    "today_hours": 0,
                    "current_login_time": agent.login_time,
                    "login_sessions": [],
                })
        return {"agents": agents_out, "live_sessions": self.list_sessions()}

    def get_session_full_messages(self, session_id: str) -> list:
        """Return the complete message history for a session (all roles)."""
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp, "meta": m.meta}
            for m in session.history
        ]

    def get_all_sessions_for_super_admin(self) -> list:
        """Metadata list of ALL sessions for the super-admin sessions browser."""
        out = []
        for s in sorted(self._sessions.values(), key=lambda x: x.created_at, reverse=True):
            first_user = next((m.content[:80] for m in s.history if m.role == "user"), "")
            has_agent = any(m.role == "agent" for m in s.history)
            out.append({
                "session_id": s.session_id,
                "created_at": s.created_at,
                "status": s.status,
                "message_count": len(s.history),
                "title": first_user or f"Session #{s.session_id}",
                "has_agent": has_agent,
                "tone": s.tone,
                "active_agent": s.active_agent,
                "awaiting_agent_confirmation": s.awaiting_agent_confirmation,
            })
        return out

    def block_agent(self, name: str) -> bool:
        rec = self._ensure_agent_record(name)
        rec["blocked"] = True
        self._save_agent_records()
        for agent in self._agents.values():
            if agent.name == name:
                agent.blocked = True
        return True

    def unblock_agent(self, name: str) -> bool:
        rec = self._agent_records.get(name)
        if rec is None:
            return False
        rec["blocked"] = False
        self._save_agent_records()
        for agent in self._agents.values():
            if agent.name == name:
                agent.blocked = False
        return True

    async def super_admin_assign_session(self, session_id: str, agent_name: str) -> bool:
        session = self._sessions.get(session_id)
        if not session:
            return False
        target = next((a for a in self._agents.values() if a.name == agent_name and not a.blocked), None)
        if not target:
            return False
        await self._assign_agent(session, target)
        return True

    async def set_active_agent(self, session_id: str, agent_name: str) -> bool:
        """Switch which bot/agent answers this session (e.g. 'layla' -> 'ava').
        Used both when a user confirms a handoff offer and when Super Admin or
        Agent Dashboard manually reroutes a session."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.active_agent = agent_name
        session.awaiting_agent_confirmation = ""
        await self._save_session(session_id)
        await self._broadcast_sessions_update()
        return True

    async def request_agent_confirmation(self, session_id: str, agent_name: str) -> bool:
        """Marks a session as awaiting the user's yes/no confirmation before
        switching active_agent to agent_name. Does NOT change active_agent yet —
        only set_active_agent() (called after the user says yes) does that."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.awaiting_agent_confirmation = agent_name
        await self._save_session(session_id)
        await self._broadcast_sessions_update()
        return True

    async def clear_agent_confirmation(self, session_id: str) -> bool:
        """Called when the user declines the handoff offer — clears the pending
        confirmation without changing active_agent."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        session.awaiting_agent_confirmation = ""
        await self._save_session(session_id)
        await self._broadcast_sessions_update()
        return True

    @staticmethod
    def response_needs_human(response: str, sources: list, upstream_needs_human: bool | None = None) -> bool:
        """
        upstream_needs_human, when provided (not None), is
        multi_source_rag.py's own grounding-check result (the lexical
        coverage checks plus the semantic _verify_grounding() backstop) —
        it inspects the actual retrieved context against the question, not
        just the final answer text after the fact the way the phrase list
        below does.

        This closes a real gap: a confidently-generated, wrong-topic answer
        (e.g. Takaful-model content used to answer "which insurers cover
        travel to South Africa") cites real (but irrelevant) sources and
        never uses any of the canned refusal phrases below — phrase-matching
        plus "has sources, so it's probably fine" then wrongly concluded no
        handoff was needed, even though multi_source_rag.py's own grounding
        checks had the information to know better. OR the two signals
        together so upstream can only ADD a missing handoff trigger, never
        remove one the phrase check already found correctly on its own.

        When upstream_needs_human is None (not passed), behaves exactly as
        before — this is the path for any caller with no upstream grounding
        signal to provide.
        """
        phrases = [
            # Explicit can't-answer phrases
            "don't have information",
            "don't have that",
            "not in my knowledge",
            "not in the documents",
            "can't find",
            "couldn't find",
            "no information",
            "not sure about",
            "can't answer",
            "cannot answer",
            "don't know",
            "outside my knowledge",
            # Handoff canned messages (must trigger even when sources exist)
            "let me get one of our agents",
            "let me get a human agent",
            "get one of our agents on it",
            "connect you with a human",
            # AI used general knowledge fallback (label added by multi_source_rag)
            "general knowledge (not from your uploaded documents)",
            "not from your uploaded documents",
            "not in the uploaded documents",
            "not covered in",
            "not available in",
        ]
        lower = response.lower()
        result = any(p in lower for p in phrases)
        # Only skip on sources when there is no explicit handoff phrase in the text.
        # If the text itself says "let me get an agent", sources are irrelevant —
        # the answer was replaced by the handoff message and must trigger a popup.
        if not result and sources:
            result = False
        if upstream_needs_human is not None:
            result = result or upstream_needs_human
        logger.info("response_needs_human=%s | sources=%d | upstream=%s | response_snippet=%r",
                    result, len(sources), upstream_needs_human, response[:120])
        return result


hub = AgentHub()


def _send_email_sync(session_id: str, history, unanswerable_query: str):
    """Synchronous wrapper — runs in a thread via asyncio.to_thread."""
    try:
        from email_utils import send_escalation_email
        send_escalation_email(session_id, history, unanswerable_query)
    except Exception:
        import logging
        logging.getLogger(__name__).exception("Email send failed for session %s", session_id)