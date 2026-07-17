from sqlalchemy import Column, Integer, String, Text, DateTime
from sqlalchemy.sql import func
from travel_bot.core.database import Base


class FieldExtractionLog(Base):
    """
    One row per field the moment it's FIRST captured for a session — this is
    what actually answers "when, and after what message, did we get the
    user's X" rather than the sessions table, which only ever holds the
    current merged snapshot (extracted_data) and has no memory of how or
    when each piece arrived.

    Written from app/routers/chat.py, right after `newly_filled` is computed
    (chat flow) or right after a document's fields are merged (upload flow) —
    those are the only two places new field values ever enter a session, so
    that's the only place this needs to be called from.
    """
    __tablename__ = "field_extraction_log"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    field_name = Column(String, index=True)
    field_value = Column(Text)
    # The user message (or "📎 Uploaded document: <filename>") that produced
    # this value — lets you trace a value back to exactly what triggered it.
    source_message = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())