"""
Human-agent handoff hub — manages chat sessions and WebSocket connections
for live agent monitoring and real-time conversation takeover.

Sessions are persisted to sessions_data.json (mounted volume) so history
survives backend restarts and agent logouts.
"""
import asyncio
import json
import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List, Set

logger = logging.getLogger(__name__)

from fastapi import WebSocket

_HERE = os.path.dirname(os.path.abspath(__file__))
_SESSIONS_FILE = os.path.join(_HERE, "sessions_data.json")
_AGENT_ACTIVITY_FILE = os.path.join(_HERE, "agent_activity.json")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M")

def _now_full() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


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
        self._load_sessions()
        self._load_agent_records()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_sessions(self):
        if not os.path.exists(_SESSIONS_FILE):
            return
        try:
            with open(_SESSIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sid, s in data.items():
                session = ChatSession(
                    session_id=sid,
                    status="ai",  # always "ai" on load; no agents connected yet
                    created_at=s.get("created_at", _now_full()),
                    tone=s.get("tone", "neutral"),
                    handoff_exhausted=s.get("handoff_exhausted", False),
                    email_sent=s.get("email_sent", False),
                )
                for m in s.get("history", []):
                    session.history.append(ChatMessage(
                        role=m["role"],
                        content=m["content"],
                        timestamp=m.get("timestamp", ""),
                        meta=m.get("meta", {}),
                    ))
                self._sessions[sid] = session
        except Exception:
            pass

    def _save_sessions(self):
        try:
            data = {}
            for sid, s in self._sessions.items():
                data[sid] = {
                    "created_at": s.created_at,
                    "tone": s.tone,
                    "handoff_exhausted": s.handoff_exhausted,
                    "email_sent": s.email_sent,
                    "history": [
                        {"role": m.role, "content": m.content, "timestamp": m.timestamp, "meta": m.meta}
                        for m in s.history
                    ],
                }
            with open(_SESSIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

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

    def create_session(self) -> str:
        sid = uuid.uuid4().hex[:8]
        self._sessions[sid] = ChatSession(session_id=sid)
        self._save_sessions()
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
                    "message": "This conversation was cleared by an agent.",
                })
            except Exception:
                pass
        del self._sessions[session_id]
        self._save_sessions()
        await self._broadcast_sessions_update()
        return True

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        return self._sessions.get(session_id)

    def get_or_create_session(self, session_id: str) -> "ChatSession":
        """Return existing session or create one on the fly (handles backend restarts)."""
        if session_id not in self._sessions:
            self._sessions[session_id] = ChatSession(session_id=session_id)
            self._save_sessions()
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
        self._save_sessions()
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
                self._save_sessions()
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
        self._save_sessions()

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

    async def _handoff_timeout(self, session_id: str, unanswerable_query: str):
        """Called after _HANDOFF_TIMEOUT seconds if no agent accepted the popup."""
        await asyncio.sleep(_HANDOFF_TIMEOUT)
        if session_id not in self._pending_handoffs:
            return  # already accepted — task was cancelled
        self._pending_handoffs.pop(session_id, None)
        session = self._sessions.get(session_id)
        _timeout_msg = {
            "type": "handoff_timeout",
            "message": "No agent is available right now. We've emailed our support team and someone will reach out to you soon.",
        }
        if session and session.status == "waiting":
            session.status = "ai"
            session.handoff_exhausted = True
            session.email_sent = True
            # Tag the triggering user message so the super-admin "Agent Only" view can flag it
            for m in reversed(session.history):
                if m.role == "user":
                    m.meta["escalation_sent"] = True
                    break
            delivered = False
            if session.user_ws:
                try:
                    await session.user_ws.send_json(_timeout_msg)
                    delivered = True
                except Exception:
                    pass
            if not delivered:
                # WS disconnected — buffer so it's sent on next reconnect
                session.pending_ws_message = _timeout_msg
            self._save_sessions()
        # Send escalation email in a thread so we don't block the event loop
        import asyncio as _aio
        history_snapshot = list(session.history) if session else []
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
        # Tag the triggering user message so the super-admin "Agent Only" view can flag it
        for m in reversed(session.history):
            if m.role == "user":
                m.meta["escalation_sent"] = True
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
        self._save_sessions()

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
        if session:
            session.email_sent = True
            # Tag the triggering user message so the super-admin "Agent Only" view can flag it
            for m in reversed(session.history):
                if m.role == "user":
                    m.meta["escalation_sent"] = True
                    break
            self._save_sessions()
        history_snapshot = list(session.history) if session else []
        await _aio.to_thread(_send_email_sync, session_id, history_snapshot, unanswerable_query)
        await self._broadcast_sessions_update()

    async def _assign_agent(self, session: "ChatSession", agent: "HumanAgent"):
        # ── Release the agent's existing session first (if any) ────────────────
        if agent.active_session and agent.active_session != session.session_id:
            old_session = self._sessions.get(agent.active_session)
            if old_session and old_session.agent_id == agent.agent_id:
                old_session.status = "ai"
                old_session.agent_id = None
                self._save_sessions()
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
        self._save_sessions()
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
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in session.history[-100:]
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
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in session.history[-100:]
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
                self._save_sessions()
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
            self._save_sessions()
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
        self._save_sessions()
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

    async def user_message_to_agent(self, session_id: str, content: str):
        """Log a user message during human-agent mode and broadcast to monitoring agents."""
        session = self._sessions.get(session_id)
        if not session:
            return
        msg = ChatMessage(role="user", content=content)
        session.history.append(msg)
        self._save_sessions()
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
        payload = {
            "type": "new_message",
            "session_id": session_id,
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp,
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
        for agent in list(self._agents.values()):
            try:
                await agent.ws.send_json({"type": "sessions_update", "sessions": sessions})
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

    @staticmethod
    def response_needs_human(response: str, sources: list) -> bool:
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
            logger.debug("response_needs_human=False (has sources, no handoff phrase)")
            return False
        logger.info("response_needs_human=%s | sources=%d | response_snippet=%r",
                    result, len(sources), response[:120])
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
