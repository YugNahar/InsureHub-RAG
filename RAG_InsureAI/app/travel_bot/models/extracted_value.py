from sqlalchemy import Column, Integer, String, Text, DateTime, UniqueConstraint
from sqlalchemy.sql import func
from travel_bot.core.database import Base


class ExtractedValue(Base):
    """
    The CURRENT value of one field for one request — one row per
    (request_id, field), upserted as values are confirmed. This is the
    resumable "source of truth" your senior described: at any point, load
    every row for a request_id, compare field names against FieldDefinition
    for that request_type, and know exactly what's collected vs. missing —
    without needing the AI agent (or any particular agent) to still be
    holding conversational memory.

    NOT the same table as FieldExtractionLog (already in this codebase) —
    that one is an append-only history answering "when/from what message did
    we get field X"; this one only ever holds the latest value per field,
    upserted in place, and is what a recovering/handed-over process would
    actually query.
    """
    __tablename__ = "extracted_values"
    __table_args__ = (
        UniqueConstraint("request_id", "field", name="uq_request_id_field"),
    )

    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, index=True)
    field = Column(String, index=True)
    value = Column(Text)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())