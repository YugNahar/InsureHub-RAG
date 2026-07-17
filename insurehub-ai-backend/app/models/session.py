from sqlalchemy import Column, Integer, String, JSON, DateTime
from sqlalchemy.sql import func
from app.core.database import Base

class ChatSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, unique=True, index=True)
    # The JSON column will store the extracted details from OpenAI
    extracted_data = Column(JSON, default={})

    # Mirror of the checklist status, kept in sync on every write (see
    # sync_field_tracking() in app/routers/chat.py — the single place that
    # updates these). Lets a session's progress be queried directly
    # (dashboard, another service, a DB browser) without re-deriving it from
    # extracted_data in Python every time.
    required_fields = Column(JSON, default=list)
    obtained_fields = Column(JSON, default=list)
    missing_fields = Column(JSON, default=list)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())