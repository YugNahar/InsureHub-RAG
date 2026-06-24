"""
Human-agent handoff hub — manages chat sessions and WebSocket connections
for live agent monitoring and real-time conversation takeover.

Sessions are persisted to sessions_data.json (mounted volume) so history
survives backend restarts and agent logouts.
"""
import asyncio
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, List, Set

from fastapi import WebSocket

_HERE = os.path.dirname(os.path.abspath(__file__))
_SESSIONS_FILE = os.path.join(_HERE, "sessions_data.json")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M")

def _now_full() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


@dataclass
class ChatMessage:
    role: str   # "user" | "ai" | "agent" | "system"
    content: str
    timestamp: str = field(default_factory=_now)


@dataclass
class ChatSession:
    session_id: str
    history: List[ChatMessage] = field(default_factory=list)
    status: str = "ai"          # "ai" | "waiting" | "human"
    agent_id: Optional[str] = None
    user_ws: Optional[WebSocket] = None
    created_at: str = field(default_factory=_now_full)


@dataclass
class HumanAgent:
    agent_id: str
    name: str
    ws: WebSocket
    active_session: Optional[str] = None
    monitoring: Set[str] = field(default_factory=set)


class AgentHub:

    def __init__(self):
        self._sessions: Dict[str, ChatSession] = {}
        self._agents: Dict[str, HumanAgent] = {}
        self._load_sessions()

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
                    status="ai",  # always start as "ai" on load; no agents connected yet
                    created_at=s.get("created_at", _now_full()),
                )
                for m in s.get("history", []):
                    session.history.append(ChatMessage(
                        role=m["role"],
                        content=m["content"],
                        timestamp=m.get("timestamp", ""),
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
                    "history": [
                        {"role": m.role, "content": m.content, "timestamp": m.timestamp}
                        for m in s.history
                    ],
                }
            with open(_SESSIONS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

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
            out.append({
                "session_id": s.session_id,
                "status": s.status,
                "agent_id": s.agent_id,
                "message_count": len(s.history),
                "created_at": s.created_at,
                "last_message": last,
                "title": first_user or f"Session #{s.session_id}",
            })
        return out

    async def log_message(self, session_id: str, role: str, content: str):
        session = self._sessions.get(session_id)
        if not session:
            return
        msg = ChatMessage(role=role, content=content)
        session.history.append(msg)
        self._save_sessions()
        await self._broadcast_new_message(session_id, msg)
        await self._broadcast_sessions_update()

    # ── Agent registration ────────────────────────────────────────────────────

    def register_agent(self, agent_id: str, name: str, ws: WebSocket) -> "HumanAgent":
        agent = HumanAgent(agent_id=agent_id, name=name, ws=ws)
        self._agents[agent_id] = agent
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
                self._save_sessions()
                if session.user_ws:
                    try:
                        await session.user_ws.send_json({
                            "type": "agent_left",
                            "message": "The agent disconnected. You're back with Layla.",
                        })
                    except Exception:
                        pass
        await self._broadcast_sessions_update()

    def online_count(self) -> int:
        return len(self._agents)

    # ── Handoff ───────────────────────────────────────────────────────────────

    async def request_handoff(self, session_id: str) -> bool:
        """Returns True if a free agent was immediately assigned."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        if session.status == "human":
            return True
        free = [a for a in self._agents.values() if a.active_session is None]
        if not free:
            session.status = "waiting"
            await self._broadcast_sessions_update()
            return False
        await self._assign_agent(session, free[0])
        return True

    async def _assign_agent(self, session: "ChatSession", agent: "HumanAgent"):
        session.status = "human"
        session.agent_id = agent.agent_id
        agent.active_session = session.session_id
        agent.monitoring.add(session.session_id)
        history_payload = [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in session.history
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
        history_payload = [
            {"role": m.role, "content": m.content, "timestamp": m.timestamp}
            for m in session.history
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
        if not agent or not session:
            return
        if agent.active_session and agent.active_session != session_id:
            old = self._sessions.get(agent.active_session)
            if old:
                old.status = "ai"
                old.agent_id = None
        await self._assign_agent(session, agent)

    async def agent_release(self, agent_id: str):
        agent = self._agents.get(agent_id)
        if not agent or not agent.active_session:
            return
        session = self._sessions.get(agent.active_session)
        if session:
            session.status = "ai"
            session.agent_id = None
            self._save_sessions()
            if session.user_ws:
                try:
                    await session.user_ws.send_json({
                        "type": "agent_left",
                        "message": "The agent has stepped away. Layla is back to help!",
                    })
                except Exception:
                    pass
        agent.active_session = None
        await self._broadcast_sessions_update()

    async def agent_send_message(self, agent_id: str, session_id: str, content: str):
        agent = self._agents.get(agent_id)
        session = self._sessions.get(session_id)
        if not agent or not session:
            return
        msg = ChatMessage(role="agent", content=content)
        session.history.append(msg)
        self._save_sessions()
        await self._broadcast_new_message(session_id, msg)
        await self._broadcast_sessions_update()
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
        await self._broadcast_new_message(session_id, msg)
        await self._broadcast_sessions_update()

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

    async def _broadcast_sessions_update(self):
        sessions = self.list_sessions()
        for agent in list(self._agents.values()):
            try:
                await agent.ws.send_json({"type": "sessions_update", "sessions": sessions})
            except Exception:
                pass

    @staticmethod
    def response_needs_human(response: str, sources: list) -> bool:
        if sources:
            return False
        phrases = [
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
        ]
        lower = response.lower()
        return any(p in lower for p in phrases)


hub = AgentHub()
