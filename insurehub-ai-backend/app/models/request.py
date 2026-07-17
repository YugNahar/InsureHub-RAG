from sqlalchemy import Column, String, DateTime
from sqlalchemy.sql import func
from app.core.database import Base


class Request(Base):
    """
    One row per insurance request/conversation. request_id is the identifier
    the whole workflow revolves around — NOT the same thing as Protego's own
    session_id/client_id from create-session (those are a third party's IDs
    for a completely different system; this is ours, for our own recovery/
    handover story).

    In this codebase, request_id == the chat session_id already used
    everywhere (ChatRequest.session_id / ChatSession.session_id) — this table
    doesn't replace ChatSession, it's a parallel, request-type-agnostic view
    of the same conversation, matching the generic Request/FieldDefinition/
    ExtractedValue design your senior described (built so it isn't tied to
    "chat" or "Travel" specifically, unlike ChatSession/extracted_data).
    """
    __tablename__ = "requests"

    request_id = Column(String, primary_key=True, index=True)
    request_type = Column(String, default="Travel Insurance")
    status = Column(String, default="IN_PROGRESS")  # IN_PROGRESS | COMPLETED | ABANDONED
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())