from typing import Any, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    options: Optional[list[str]] = None
    ui: Optional[dict[str, Any]] = None
    collected_fields: Optional[dict[str, Any]] = None


class IntakeFormTraveller(BaseModel):
    first_name: str = ""
    last_name: str = ""
    date_of_birth: str = ""


class IntakeFormRequest(BaseModel):
    """Structured, single-shot alternative to answering the QUOTING-phase
    questions one at a time in chat — see submit_intake_form_endpoint."""
    session_id: str
    first_name: str = ""
    last_name: str = ""
    email: str = ""
    mobile_number: str = ""
    coverage_type: str = ""
    destination: str = ""
    departure: str = ""
    start_date: str = ""
    end_date: str = ""
    plan_type: str = ""
    cover_type: str = ""
    date_of_birth: str = ""
    additional_travellers: list[IntakeFormTraveller] = []
    marketing_consent: bool = False


class EmailQuoteRequest(BaseModel):
    """Payload for the 'Email me this quote' button — the quote data itself
    is never sent by the client, it's re-read from the session's own state
    server-side (see email_quote_endpoint) to avoid trusting a possibly
    stale/tampered client-side snapshot."""
    session_id: str
    email: str