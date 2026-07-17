from sqlalchemy import Column, Integer, String, Boolean, Text, UniqueConstraint
from app.core.database import Base


class FieldDefinition(Base):
    """
    Predefined metadata for every field that COULD be needed for a given
    request_type (e.g. "Travel Insurance"). This is seeded once (see
    seed_field_definitions.py) and rarely changes at runtime — it's the
    "checklist template" that ExtractedValue rows get compared against to
    figure out what's missing for a given request.
    """
    __tablename__ = "field_definitions"
    __table_args__ = (
        UniqueConstraint("request_type", "field_key", name="uq_request_type_field_key"),
    )

    id = Column(Integer, primary_key=True, index=True)
    request_type = Column(String, index=True)      # e.g. "Travel Insurance"
    field_key = Column(String, index=True)          # e.g. "passport_number" — stable, code-facing
    display_name = Column(String)                   # e.g. "Passport Number" — human-facing
    data_type = Column(String)                      # "string" | "date" | "number" | "enum" | "list"
    is_required = Column(Boolean, default=True)
    validation_rules = Column(Text)                 # JSON-encoded string, e.g. '{"format": "YYYY-MM-DD"}'
    display_order = Column(Integer)