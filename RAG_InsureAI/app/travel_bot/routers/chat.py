from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from datetime import datetime, timedelta
import random
import re
import json
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.orm.attributes import flag_modified
from travel_bot.schemas.chat import ChatRequest, ChatResponse, IntakeFormRequest, EmailQuoteRequest
from travel_bot.models.message import Message
from travel_bot.models.session import ChatSession
from travel_bot.models.field_log import FieldExtractionLog
from travel_bot.models.request import Request
from travel_bot.models.extracted_value import ExtractedValue
from travel_bot.core.database import get_db
from travel_bot.services.llm_service import LLMService
from travel_bot.services.quote_service import QuoteService
from travel_bot.services.currency_service import format_dual_currency
from travel_bot.schemas.travel import (
    validate_coverage_destination,
    infer_coverage_type_from_destination,
    normalize_country,
    normalize_date,
    COVERAGE_TYPE_OPTIONS,
    COVERAGE_TYPE_LOCKS,
    COVERAGE_TYPE_RULES,
    MULTI_TRAVELLER_COVER_TYPES,
    MIN_TRAVELLERS_FOR_MULTI,
    MAX_TRAVELLERS_TOTAL,
)

router = APIRouter(prefix="/api/chat", tags=["Chat"])
llm_service = LLMService()

# Deliberately permissive (accepts business/personal/education addresses
# alike, not a strict RFC 5322 pattern) but requires a real-looking TLD —
# used both for the intake-form backstop and the email-quote endpoint.
_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[a-zA-Z]{2,}$")

# coverage_type (the product / destination rules) and cover_type (who's
# insured) are deliberately both here as SEPARATE required fields — they are
# not interchangeable. See app/schemas/travel.py for the full rule table.
REQUIRED_FIELDS = [
    "first_name", "last_name", "email", "mobile_number",
    "coverage_type", "destination", "departure", "start_date", "end_date",
    "plan_type", "cover_type", "date_of_birth"
]

FIELD_LABELS = {
    "first_name": "First name",
    "last_name": "Last name",
    "email": "Email",
    "mobile_number": "Mobile number",
    "coverage_type": f"Coverage type ({' / '.join(COVERAGE_TYPE_OPTIONS)})",
    "destination": "Destination country",
    "departure": "Country you're travelling from",
    "start_date": "Trip start date",
    "end_date": "Trip end date",
    "plan_type": "Plan type (Single Trip / Annual Multi-Trip)",
    "cover_type": "Cover type — who's insured (Individual / Group / Family)",
    "date_of_birth": "Date of birth",
}

def _short_field_label(field: str) -> str:
    """FIELD_LABELS' parenthetical option lists ("(Hajj and Umrah / UAE
    Inbound / ...)") and the cover_type em-dash clarifier exist to help
    the user pick a value while a field is still being ASKED for — they
    read as clutter (and previously got wrongly lowercased into "hajj and
    umrah" / "uae inbound") when just naming a field that's already been
    successfully captured. Strips everything from the first "(" or "—"
    onward, leaving just the bare field name for that use.
    """
    return re.split(r"\s*[(—]", FIELD_LABELS[field], maxsplit=1)[0]


# Conversational phrasing for asking each field, used instead of dumping the
# raw FIELD_LABELS bullet list — see build_friendly_missing_prompt(). Each
# entry is a complete, politely-phrased, capitalized question (not a
# lowercase mid-sentence fragment) — confirmed live that a human agent
# leads with "Can you please..."/"Could you..." rather than a bare
# "what's your X?", and reads more natural capitalized even after an
# em-dash lead-in like "Just one more thing — ".
FIELD_QUESTIONS = {
    "first_name": "Could you please share your first name?",
    # Only ever rendered standalone now — build_friendly_missing_prompt
    # collapses first_name+last_name into one "full name" ask when both are
    # missing together, so this no longer needs the "and" that assumed it
    # would always follow the first_name question.
    "last_name": "Could you also share your last name?",
    "email": "Could you please share the best email address to reach you on?",
    "mobile_number": "Could you please share your mobile number?",
    "coverage_type": "Could you let me know if this is for Hajj & Umrah, UAE Inbound, GCC Countries, Schengen, or Worldwide cover?",
    "destination": "Which country will you be travelling to?",
    "departure": "And which country will you be travelling from?",
    "start_date": "When would you like the trip to start?",
    "end_date": "And when does it end?",
    "plan_type": "Would you like a Single Trip plan, or Annual Multi-Trip cover?",
    "cover_type": "Are you travelling solo, or with a group or family?",
    "date_of_birth": "Could you please share your date of birth?",
}

# Discrete-choice fields — used to attach tappable quick-reply buttons
# instead of requiring free text, for whichever single field is being asked
# next. Fields not listed here (name, email, dates, etc.) stay free text.
FIELD_OPTIONS_MAP = {
    "cover_type": ["Individual", "Group", "Family"],
    "plan_type": ["Single Trip", "Annual Multi-Trip"],
    "coverage_type": ["Hajj and Umrah", "UAE Inbound", "GCC Countries", "Schengen", "Worldwide"],
}


# ── In-flow Q&A ("what is coverage type?", "how do I fill this in?") ────────
# Same two-stage discipline as agent_router.interrupt's _GENERAL_QUESTION_RE:
# a fast regex catches the obvious cases for free; only a question-shaped
# message with no obvious vocab hit pays for an LLM call. Answered entirely
# by Ava's own LLM (llm_service, Groq-backed) — never routed to Layla, since
# Layla's general RAG knowledge base doesn't document Ava's own vocabulary
# (the 5 coverage types, etc.) and answers these badly.
_AVA_PROCESS_VOCAB_RE = re.compile(
    r"\b(?:coverage\s*type|cover\s*type|plan\s*type|quotation|quote|policy|"
    r"add-?ons?|destination|departure|hajj|umrah|schengen|worldwide|"
    r"gcc\b|uae\s*inbound|mobile\s*number|date\s*of\s*birth|passport|"
    r"emirates\s*id|this\s*form|this\s*field)\b",
    re.IGNORECASE,
)
_QUESTION_SHAPE_RE = re.compile(
    r"^\s*(?:what|how|why|which|when|where|who|can|could|does|do|is|are)\b|\?\s*$",
    re.IGNORECASE,
)


def is_ava_process_question(message: str) -> bool:
    """Safe default is False (fall through to the normal phase machine) on
    any ambiguity — a missed question just gets the existing canned/field-
    extraction behavior, same as today; a false positive would incorrectly
    hijack a real form answer, which is the worse failure mode."""
    text = message.strip()
    if not text:
        return False
    has_vocab = bool(_AVA_PROCESS_VOCAB_RE.search(text))
    has_question_shape = bool(_QUESTION_SHAPE_RE.search(text))
    if has_vocab and has_question_shape:
        return True
    if not has_question_shape or len(text.split()) <= 3:
        return False
    return _classify_ava_process_question_llm(text)


def _classify_ava_process_question_llm(message: str) -> bool:
    # Confirmed live: a vague "is this about the quote process?" framing
    # over-triggers on ANY insurance-shaped question, including a totally
    # unrelated one ("What is motor insurance?" -> "yes") — this must stay
    # narrow, since a false positive here hijacks a question that should
    # fall through to Layla (see api.py's interruption-gate override), and
    # a missed one just falls back to today's existing behavior. Explicit
    # positive/negative examples fixed both misses below.
    try:
        llm = llm_service._get_llm(temperature=0.0, max_tokens=5)
        prompt = f"""A user is filling out a travel insurance quote form, or reviewing a
quote, from Ava — an assistant that ONLY handles this one specific travel-insurance quote
flow (coverage types: Hajj and Umrah, UAE Inbound, GCC Countries, Schengen, Worldwide;
trip dates, destination/departure countries, traveller details, the quotes shown).

Classify the MESSAGE as exactly one of:
"yes" — a question specifically about THIS travel-quote process: what a form field or
  coverage-type option means, how to fill something in, or something about a quote
  Ava has already shown.
"no" — anything else: a question about a DIFFERENT insurance product (motor, home,
  health, life, etc.), a general insurance concept not specific to this travel flow, or
  a request unrelated to getting a travel quote.

Examples:
"what is coverage type" -> yes
"how do I fill this in" -> yes
"what does option 1 include" -> yes
"why do you need my passport" -> yes
"what is motor insurance?" -> no
"does home insurance cover flood damage?" -> no
"what is a deductible?" -> no

MESSAGE: {message}

Reply with ONLY "yes" or "no" — nothing else."""
        resp = llm.invoke(prompt)
        answer = (getattr(resp, "content", "") or "").strip().lower()
        return answer.startswith("yes")
    except Exception:
        return False


def answer_ava_process_question(message: str, state: dict) -> str:
    coverage_lines = []
    for ct, rule in COVERAGE_TYPE_RULES.items():
        dest = ", ".join(rule["destination_options"]) if rule.get("destination_options") else "any country"
        dep = ", ".join(rule["departure_options"]) if rule.get("departure_options") else "any country"
        coverage_lines.append(f"- {ct}: destination = {dest}; departure = {dep}")
    coverage_text = "\n".join(coverage_lines)

    field_lines = "\n".join(f"- {FIELD_LABELS.get(f, f)}: {q}" for f, q in FIELD_QUESTIONS.items())

    quotes = state.get("available_quotes") or []
    quotes_text = ("\n" + quotes_summary_for_email(state)) if quotes else " (none fetched yet)"

    prompt = f"""You are Ava, a travel-insurance quote assistant for InsureHub. A user asked
a question about the quote process or a quote you've already shown them — answer it
directly, clearly, and briefly (2-4 sentences, plain text, no markdown headers).

COVERAGE TYPES (destination/departure rules):
{coverage_text}

FORM FIELDS:
{field_lines}

CURRENTLY SHOWN QUOTES:{quotes_text}

USER'S QUESTION: {message}

Answer only using the information above. If the question isn't something you can answer
from this context, say you're not sure and suggest they ask their question a different way."""

    try:
        llm = llm_service._get_llm(temperature=0.2, max_tokens=300)
        resp = llm.invoke(prompt)
        answer = (getattr(resp, "content", "") or "").strip()
        return answer or "Sorry, I couldn't come up with an answer to that — could you rephrase?"
    except Exception:
        return "Sorry, I'm having trouble answering that right now — could you try again in a moment?"


def determine_reply_options(state: dict, reply_text: str) -> list:
    """
    Best-effort guess at which tappable quick-reply options (if any) match
    the question `reply_text` just asked, so the frontend can render buttons
    for simple discrete-choice questions instead of requiring the user to
    type. Returns [] for open-ended questions (free text expected) — this is
    purely additive UI sugar, never enforced server-side, so a wrong/missed
    guess here just means the user types instead of taps, nothing breaks.
    """
    if not reply_text:
        return []

    if state.get("_awaiting_marketing_consent"):
        return ["Yes", "No"]

    if state.get("phase") == "CHOOSING" and "reply with a number" in reply_text.lower():
        n = min(len(state.get("available_quotes", [])), 5)
        return [str(i) for i in range(1, n + 1)]

    if state.get("phase") == "QUOTING":
        missing = [f for f in REQUIRED_FIELDS if not state.get(f)]
        if len(missing) == 1 and missing[0] in FIELD_OPTIONS_MAP:
            return FIELD_OPTIONS_MAP[missing[0]]

    return []


def is_quote_checklist_complete(data: dict) -> bool:
    for field in REQUIRED_FIELDS:
        if not data.get(field) or str(data.get(field)).strip() == "":
            return False
    return True


def merge_extracted_fields(state: dict, extracted: dict) -> dict:
    """Only write fields that actually have a value — never blank out existing
    data. Also never OVERWRITE an already-filled field: the extraction
    schema requires all fields on every call (structured-output contract),
    even though the prompt only asks the model to focus on whichever fields
    are still missing — so for a field already captured, a non-empty value
    coming back here is the model guessing to satisfy the schema, not real
    signal from this message. Confirmed live: "Schengen" correctly captured,
    then a later message about dates alone came back with coverage_type
    hallucinated as "Worldwide" and silently clobbered the real answer."""
    for key, val in (extracted or {}).items():
        if state.get(key):
            continue
        if val and str(val).strip() != "":
            val = str(val).strip()
            if key in ("destination", "departure"):
                val = normalize_country(val)
            elif key in ("start_date", "end_date", "date_of_birth"):
                val = normalize_date(val)
            state[key] = val
    return state


def apply_coverage_type_locks(state: dict) -> None:
    """Once coverage_type is known, auto-fills destination and/or departure
    for whichever of those COVERAGE_TYPE_RULES restricts to exactly one
    valid value (see COVERAGE_TYPE_LOCKS) — e.g. "UAE Inbound" always means
    destination="United Arab Emirates", "Hajj and Umrah" always means
    departure="United Arab Emirates" too. Never overwrites an already-set
    value (same "don't clobber real signal" rule merge_extracted_fields
    follows) — this only fills in a field that's still genuinely blank."""
    locked = COVERAGE_TYPE_LOCKS.get(state.get("coverage_type", ""), {})
    for field, value in locked.items():
        if not state.get(field):
            state[field] = value


def _format_date_human(value: str) -> str:
    """'2026-07-22' -> '22 Jul 2026' for display. Falls back to the raw
    string for anything not in the expected ISO shape — display-only, so
    never worth raising over."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%d %b %Y")
    except (ValueError, TypeError):
        return value or ""


def _join_natural(items: list) -> str:
    """['a'] -> 'a'; ['a','b'] -> 'a and b'; ['a','b','c'] -> 'a, b, and c'."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def build_obtained_summary(state: dict) -> str:
    """A natural-language recap of everything captured so far — used right
    after a document upload, where the user hasn't seen any of this yet and
    benefits from a full picture rather than a raw field dump."""
    parts = []

    name = " ".join(p for p in [state.get("first_name"), state.get("last_name")] if p)
    if name:
        dob = state.get("date_of_birth")
        parts.append(name + (f" (born {_format_date_human(dob)})" if dob else ""))

    contact_bits = [b for b in [state.get("email"), state.get("mobile_number")] if b]
    if contact_bits:
        parts.append(_join_natural(contact_bits))

    if state.get("destination"):
        trip = f"heading to {state['destination']}"
        if state.get("start_date") or state.get("end_date"):
            trip += f" from {_format_date_human(state.get('start_date'))} to {_format_date_human(state.get('end_date'))}"
        parts.append(trip)

    product_bits = [b for b in [state.get("plan_type"), state.get("coverage_type")] if b]
    if product_bits:
        parts.append(_join_natural(product_bits) + " cover" if state.get("coverage_type") else _join_natural(product_bits))

    if not parts:
        return ""
    return "I've pulled these details from your document: " + "; ".join(parts) + "."


_GREETING_RE = re.compile(
    r"\b(hi|hello|hey|hiya|howdy|yo|good\s*(morning|afternoon|evening)|greetings)\b",
    re.IGNORECASE,
)

# Stricter than _GREETING_RE (substring match, used only for ack tone) —
# this only matches when the ENTIRE message is a greeting and nothing else,
# e.g. "Hi" or "Hey there!". "Hi, I'm John" or "Hello, going to France"
# correctly falls through to the real extraction call below since there's
# actual data alongside the greeting.
_GREETING_ONLY_RE = re.compile(
    r"^\s*(hi|hello|hey|hiya|howdy|yo|good\s*(morning|afternoon|evening)|greetings)[\s!.,]*$",
    re.IGNORECASE,
)


def is_greeting_only(message: str) -> bool:
    return bool(_GREETING_ONLY_RE.match(message or ""))

# A few natural variants so a real back-and-forth doesn't sound like the same
# canned line every time someone says hi — a human agent wouldn't repeat
# themselves verbatim either.
_GREETING_RESPONSES = [
    "Hey there! ",
    "Hi! Great to hear from you. ",
    "Hello! Happy to help. ",
]


def is_greeting(message: str) -> bool:
    return bool(_GREETING_RE.search(message or ""))


def _capitalize_first(s: str) -> str:
    """Uppercases just the first character, leaving the rest untouched.
    str.capitalize() also lowercases every other character, which would
    wrongly mangle proper nouns/acronyms already correctly cased inside
    these question fragments (e.g. 'is this for Hajj and Umrah, UAE
    Inbound, ...' -> 'Uae inbound' with .capitalize()).
    """
    return s[0].upper() + s[1:] if s else s


def build_friendly_missing_prompt(state: dict, newly_filled: list = None, user_message: str = "") -> str:
    """Conversational, incremental replacement for the old full bullet-list
    prompt: acknowledges whatever was JUST captured this turn (if anything),
    then asks for at most 2 remaining fields at once instead of the whole
    checklist — closer to how a human agent actually asks."""
    newly_filled = newly_filled or []
    missing = [f for f in REQUIRED_FIELDS if not state.get(f)]

    ack = ""
    if is_greeting(user_message):
        # Confirmed live: a bare "Hi"/"Hello" got zero acknowledgment before —
        # straight into "what's your first name?" with no greeting back,
        # which is exactly what made this feel like a form, not an agent.
        ack = random.choice(_GREETING_RESPONSES)
    if "cover_type" in newly_filled and state.get("cover_type") in MULTI_TRAVELLER_COVER_TYPES:
        ack += f"That's exciting — travelling as a {state['cover_type'].lower()}! "
    elif newly_filled:
        # Same "atomic" collapse as the question side below (full name /
        # contact details): confirmed live that "Thanks for sharing your
        # first name and last name!" read just as robotic as the stitched
        # question did, so any paired fields get named as one thing here
        # too rather than spelled out separately.
        _collapse_pairs = [
            (("first_name", "last_name"), "full name"),
            (("email", "mobile_number"), "contact details"),
        ]
        labels = []
        remaining = list(newly_filled)
        for pair, combined_label in _collapse_pairs:
            if all(f in remaining for f in pair):
                labels.append(combined_label)
                remaining = [f for f in remaining if f not in pair]
        labels.extend(_short_field_label(f).lower() for f in remaining)
        captured = _join_natural(labels)
        ack += f"Thanks for sharing your {captured}! "

    if not missing:
        return ack.strip() or "Got it, thank you!"

    to_ask = missing[:2]
    # first_name + last_name is really one "what's your name" question, not
    # two separate ones — unlike start_date/end_date (genuinely two distinct
    # pieces of info even in casual speech), a name is atomic. Confirmed
    # live: "What's your first name? And your last name?" read as two
    # separate questions stitched together, not how a human agent asks.
    # REQUIRED_FIELDS always lists first_name before last_name, so this
    # exact-order check is reliable whenever both are still missing.
    if to_ask == ["first_name", "last_name"]:
        frags = ["Can you please provide your full name?"]
    # Same reasoning applies to email + mobile_number: stacking "What's the
    # best email to reach you on? Could you share your mobile number?" back
    # to back read like a form, not a person. Collapse into one warm
    # "contact details" ask instead. REQUIRED_FIELDS lists email before
    # mobile_number, so this exact-order check is reliable too.
    elif to_ask == ["email", "mobile_number"]:
        frags = ["Could you please share your contact details — your email address and mobile number?"]
    else:
        frags = [FIELD_QUESTIONS.get(f, FIELD_LABELS[f] + "?") for f in to_ask]

    # "Just one more thing" is only honest when this ask covers literally
    # everything left — either it's the single final field, or the
    # pair-collapse above combined the last two remaining fields into one
    # question. `len(frags) == 1` on its own isn't enough: it also fires
    # when the NEXT TWO of a much longer missing list happen to be a
    # collapsible pair (e.g. email+mobile_number are next but
    # coverage_type/destination/... still remain after) — confirmed live
    # this reads as a broken promise once the very next turn asks yet
    # another question. `len(missing) <= 2` guarantees to_ask already
    # covers every remaining field, so the collapsed fragment really is
    # the last ask.
    if len(missing) <= 2 and len(frags) == 1:
        return f"{ack}Just one more thing — {frags[0]}"

    # A second question fragment always starts a fresh sentence after the
    # first one's "?", regardless of context — capitalize it unconditionally.
    if len(frags) > 1:
        frags[1] = _capitalize_first(frags[1])

    if len(missing) == 2 and len(frags) == 2:
        return f"{ack}Almost there, just two more things — " + " ".join(frags)

    # No lead-in phrase here (unlike the two cases above) — the first
    # fragment is either the very start of the reply or follows a
    # full-stop-ending ack ("Hi! Great to hear from you. "). FIELD_QUESTIONS
    # entries are already capitalized, but this stays as a defensive
    # guarantee for the FIELD_LABELS fallback branch too.
    frags[0] = _capitalize_first(frags[0])
    return f"{ack}" + " ".join(frags)


def build_missing_fields_reply(state: dict, preamble: str = "", newly_filled: list = None, user_message: str = "") -> str:
    """Kept as the single entry point call sites use, now delegating to the
    friendly incremental prompt instead of a bullet-list dump. `preamble`
    (e.g. after a document upload) is prepended as-is; pass a doc summary in
    via `preamble` when you want the full human recap shown."""
    return preamble + build_friendly_missing_prompt(state, newly_filled, user_message)


def sync_field_tracking(db_session: ChatSession, state: dict) -> None:
    """Mirrors the current checklist status onto the session's tracking
    columns, so a session's progress can be queried directly instead of
    re-derived from extracted_data every time. Call this right before every
    commit that writes `state` into db_session.extracted_data.

    obtained_fields is a {field_name: value} map (not just names) so the
    actual submitted values are visible without cross-referencing
    extracted_data. missing_fields stays a plain list of names, since a
    missing field has no value to show."""
    db_session.required_fields = REQUIRED_FIELDS
    db_session.obtained_fields = {f: state[f] for f in REQUIRED_FIELDS if state.get(f)}
    db_session.missing_fields = [f for f in REQUIRED_FIELDS if not state.get(f)]


# Maps this codebase's internal field keys to the canonical field_key naming
# in FieldDefinition (seeded via seed_field_definitions.py). first_name/
# last_name are combined into a single "full_name" value instead of being
# listed here — see sync_extracted_values below.
FIELD_KEY_MAP = {
    "email": "email_address",
    "destination": "destination_country",
    "departure": "country_of_residence",
    "start_date": "departure_date",
    "end_date": "return_date",
    "plan_type": "trip_type",
    # mobile_number, date_of_birth, coverage_type, cover_type already match
    # the canonical field_key naming 1:1, so no mapping needed for those.
}


def upsert_extracted_value(db: DBSession, request_id: str, field: str, value: str) -> None:
    row = db.query(ExtractedValue).filter(
        ExtractedValue.request_id == request_id, ExtractedValue.field == field
    ).first()
    if row:
        row.value = value
    else:
        db.add(ExtractedValue(request_id=request_id, field=field, value=value))


def sync_extracted_values(db: DBSession, request_id: str, state: dict, fields: list) -> None:
    """
    Upserts ExtractedValue rows (the resumable "current value" store) for
    whichever of `fields` are present in `state`, translated to the senior's
    canonical field_key naming. Called alongside FieldExtractionLog writes —
    that table is the append-only "when did we get X" history; this is the
    "what do we currently have" snapshot a recovering/handed-over process
    would actually query.
    """
    full_name = f"{state.get('first_name', '')} {state.get('last_name', '')}".strip()
    if full_name:
        upsert_extracted_value(db, request_id, "full_name", full_name)

    for internal_key in fields:
        if internal_key in ("first_name", "last_name"):
            continue  # combined into full_name above
        value = state.get(internal_key)
        if value:
            mapped_key = FIELD_KEY_MAP.get(internal_key, internal_key)
            upsert_extracted_value(db, request_id, mapped_key, str(value))


def check_coverage_consistency(state: dict):
    """
    Returns (bad_field, message) if coverage_type/destination/departure
    conflict with the rules in app/schemas/travel.py (e.g. coverage_type=
    'GCC Countries' but destination='France') — bad_field names exactly the
    field the mismatch is actually about, so callers can clear just that one
    rather than assuming it's always destination (confirmed live: a
    hallucinated departure value, e.g. a date extracted into the departure
    field when none was ever mentioned, was wiping out a correctly-extracted
    destination as collateral damage). Returns (None, None) if there's
    nothing to check yet (fields still missing) or everything's consistent.
    """
    coverage_type = state.get("coverage_type", "")
    destination = state.get("destination", "")
    departure = state.get("departure", "")
    if not coverage_type or not destination:
        return None, None
    bad_field, message = validate_coverage_destination(coverage_type, destination, departure)
    return (bad_field, message) if bad_field else (None, None)


def total_travellers(state: dict) -> int:
    """1 primary traveller + however many companions have been collected so far."""
    return 1 + len(state.get("additional_travellers", []))


def travellers_requirement_met(state: dict) -> bool:
    """Individual policies always satisfy this with just the primary traveller."""
    if state.get("cover_type") not in MULTI_TRAVELLER_COVER_TYPES:
        return True
    return total_travellers(state) >= MIN_TRAVELLERS_FOR_MULTI


_DONE_KEYWORDS = [
    "done", "no more", "that's all", "thats all", "that's it", "thats it",
    "nothing else", "proceed", "get quote", "get quotes", "finish", "finished",
    "complete", "go ahead", "just us", "that's everyone", "thats everyone",
]


def is_done_signal(message: str) -> bool:
    msg = (message or "").lower()
    return any(kw in msg for kw in _DONE_KEYWORDS)


MARKETING_CONSENT_QUESTION = (
    "One last thing — would you like to receive updates from InsureHub "
    "about quotes, products, and promotional offers?"
)

# Word-boundary matching (not plain substring) since "no" as a bare substring
# false-positives inside common words like "know" or "not interested".
_YES_WORDS = [r"\byes\b", r"\byeah\b", r"\byep\b", r"\bsure\b", r"\bok(ay)?\b", r"\bi would\b", r"\bi'd like\b"]
_NO_WORDS = [r"\bno\b", r"\bnope\b", r"\bnah\b", r"\bdon'?t\b", r"\bdo not\b", r"\bnot interested\b", r"\bskip\b", r"\bdecline\b"]


def parse_yes_no(message: str) -> str:
    """Returns 'yes', 'no', or '' if the message doesn't clearly say either —
    caller should re-ask rather than guess."""
    msg = (message or "").lower()
    if any(re.search(p, msg) for p in _NO_WORDS):
        return "no"
    if any(re.search(p, msg) for p in _YES_WORDS):
        return "yes"
    return ""


def build_traveller_prompt(state: dict, just_selected: bool = False) -> str:
    n = total_travellers(state)
    cover_type = state.get("cover_type", "")
    slots_left = MAX_TRAVELLERS_TOTAL - n

    if n < MIN_TRAVELLERS_FOR_MULTI:
        opener = f"That's lovely — travelling as a {cover_type.lower()}! " if just_selected else ""
        return (
            f"{opener}How many are travelling with you in total? Could you share the full name "
            f"and date of birth for the other{'s' if slots_left > 1 else ''} joining you "
            f"(up to {slots_left} more, {MAX_TRAVELLERS_TOTAL} travellers max)?"
        )
    return (
        f"Thanks — that's {n} traveller{'s' if n != 1 else ''} so far! "
        f"Anyone else joining (up to {slots_left} more), or just reply 'done' and I'll pull your quotes."
    )


def _quote_price(q: dict):
    """Returns the numeric amount_with_vat, or None if pricing is missing/zero
    (a quote with no real price, like the "0.0 None" case, should never be
    treated as the cheapest option just because 0 sorts first)."""
    amt = q.get("plan", {}).get("price", {}).get("amount_with_vat")
    return amt if isinstance(amt, (int, float)) and amt > 0 else None


def _fmt_amount(amount) -> str:
    """Formats a raw price/add-on amount to standard 2-decimal currency
    display with thousands separators. The quote API returns amount_with_vat
    as a bare float with whatever precision it happens to carry (confirmed
    live: "34.566", "71.232", "128.331" alongside "77.7", "105.0" in the same
    quote list) — displaying it unrounded reads as broken to a user even
    though the underlying number is real, not corrupted. Falls back to a
    plain str() for the rare non-numeric amount (matches the isinstance
    guard already used for the "covers up to" value just below)."""
    return f"{amount:,.2f}" if isinstance(amount, (int, float)) else str(amount)


def sort_quotes(quotes: list, sort_by: str = "price_asc") -> list:
    """Sorts a copy of `quotes`. Quotes with no usable price are always
    pushed to the end regardless of sort direction, and shown as "pricing
    unavailable" rather than looking like a free/cheapest option."""
    valid = [q for q in quotes if _quote_price(q) is not None]
    invalid = [q for q in quotes if _quote_price(q) is None]

    if sort_by == "price_desc":
        valid.sort(key=_quote_price, reverse=True)
    elif sort_by == "insurer_asc":
        valid.sort(key=lambda q: q.get("insurer", {}).get("name", "").lower())
    elif sort_by == "insurer_desc":
        valid.sort(key=lambda q: q.get("insurer", {}).get("name", "").lower(), reverse=True)
    else:  # "price_asc" (default) — matches the website's own default sort
        valid.sort(key=_quote_price)

    return valid + invalid


_SORT_LABELS = {
    "price_asc": "Price: Low → High",
    "price_desc": "Price: High → Low",
    "insurer_asc": "Insurer: A → Z",
    "insurer_desc": "Insurer: Z → A",
}


def parse_sort_command(message: str) -> str:
    """Returns a sort_by key if the message is asking to re-sort, else ''.
    Simple keyword matching — same style as is_done_signal/parse_yes_no."""
    msg = (message or "").lower()
    if not any(kw in msg for kw in ["sort", "cheapest", "cheaper", "expensive", "a-z", "z-a"]):
        return ""
    wants_insurer = "insurer" in msg or "provider" in msg or "name" in msg or "a-z" in msg or "z-a" in msg
    wants_desc = any(kw in msg for kw in ["high", "expensive", "desc", "z-a"])
    if wants_insurer:
        return "insurer_desc" if wants_desc else "insurer_asc"
    return "price_desc" if wants_desc else "price_asc"


def format_quotes_list(state: dict, sort_by: str = "price_asc") -> str:
    """Sorts state['available_quotes'] in place and renders the top 5 as
    numbered options — single source of truth used both for the initial
    quote reply and for re-renders after a sort command."""
    quotes = sort_quotes(state.get("available_quotes", []), sort_by)
    state["available_quotes"] = quotes
    state["quote_sort"] = sort_by

    top5 = quotes[:5]
    scores = [(_value_score(q), i) for i, q in enumerate(top5)]
    scores = [(s, i) for s, i in scores if s is not None]
    best_value_idx = max(scores, key=lambda pair: pair[0])[1] if len(scores) >= 2 else None

    # Protego always prices in AED regardless of departure country — this is
    # a DISPLAY-only conversion for a user who said they're travelling from
    # somewhere else (e.g. India), never touches what's actually quoted/
    # charged. See currency_service.format_dual_currency's own docstring.
    departure = state.get("departure", "")

    # len(quotes) is the FULL pool (e.g. 12); only top5 (max 5) are ever
    # actually listed below — the old wording ("X quote(s) found:") read as
    # a mismatch/bug when X > 5 since the message promised more than it
    # showed. Spell out "showing top N of X" whenever they differ.
    count_line = (
        f"Sorted by {_SORT_LABELS.get(sort_by, _SORT_LABELS['price_asc'])} — "
        f"showing top {len(top5)} of {len(quotes)} quote(s) found:\n"
        if len(quotes) > len(top5)
        else f"Sorted by {_SORT_LABELS.get(sort_by, _SORT_LABELS['price_asc'])} — {len(quotes)} quote(s) found:\n"
    )
    lines = [count_line]
    for i, q in enumerate(top5):
        insurer = q.get("insurer", {}).get("name", "Insurer")
        plan = q.get("plan", {}).get("name", "Standard")
        price = _quote_price(q)
        currency = q.get("plan", {}).get("price", {}).get("currency", "AED")
        if price and currency == "AED":
            price_str = format_dual_currency(price, departure)
        elif price:
            price_str = f"{_fmt_amount(price)} {currency}"
        else:
            price_str = "pricing unavailable"
        badge = " ⭐ Best value (medical coverage per AED spent)" if i == best_value_idx else ""
        lines.append(f"Option {i + 1}: {insurer} ({plan}) — {price_str}{badge}")

    lines.append(
        "\nReply with a number to select a plan, or ask me to sort by price/insurer, "
        "show add-ons for option N, or compare 1 and 2 (etc.)."
    )
    return "\n".join(lines)


def quotes_summary_for_email(state: dict) -> str:
    """Same per-option formatting as format_quotes_list (insurer, plan,
    dual-currency price) minus the "reply with a number" chat-only CTA —
    used by email_quote_endpoint so the emailed quote matches what's on
    screen. Does not re-sort — uses whatever order is currently in
    state['available_quotes'] (already sorted by the last sort command,
    or price_asc by default from try_fetch_quotes)."""
    quotes = state.get("available_quotes", [])[:5]
    departure = state.get("departure", "")
    lines = []
    for i, q in enumerate(quotes):
        insurer = q.get("insurer", {}).get("name", "Insurer")
        plan = q.get("plan", {}).get("name", "Standard")
        price = _quote_price(q)
        currency = q.get("plan", {}).get("price", {}).get("currency", "AED")
        if price and currency == "AED":
            price_str = format_dual_currency(price, departure)
        elif price:
            price_str = f"{_fmt_amount(price)} {currency}"
        else:
            price_str = "pricing unavailable"
        lines.append(f"Option {i + 1}: {insurer} ({plan}) — {price_str}")
    return "\n".join(lines)


def trip_summary_for_email(state: dict) -> str:
    parts = [
        f"{state.get('destination', '')} ({state.get('coverage_type', '')})",
        f"{_format_date_human(state.get('start_date', ''))} – {_format_date_human(state.get('end_date', ''))}",
        f"Travelling from {state.get('departure', '')}" if state.get("departure") else "",
        f"{state.get('cover_type', '')} cover, {state.get('plan_type', '')}",
    ]
    return " · ".join(p for p in parts if p.strip(" ()–"))


def parse_option_number(message: str) -> int:
    """0-based option index from things like 'option 2', 'quote 3', or a bare
    number — used for add-ons lookups. Returns -1 if none found."""
    match = re.search(r"\b(?:option|quote)?\s*#?\s*([1-5])\b", (message or "").lower())
    return int(match.group(1)) - 1 if match else -1


def parse_compare_indices(message: str) -> list:
    """0-based option indices mentioned in a 'compare 1 and 3' style message,
    deduplicated, in the order they appear."""
    seen = []
    for n in re.findall(r"\b([1-5])\b", message or ""):
        idx = int(n) - 1
        if idx not in seen:
            seen.append(idx)
    return seen


def format_addons_for_quote(quote: dict) -> str:
    """Renders the 'Add-Ons' section of one raw Protego quote object — each
    add-on item carries its own extra price (amount) alongside what it
    covers (value), distinct from the base plan price."""
    for section in quote.get("sections", []):
        if section.get("title", "").strip().lower() == "add-ons":
            items = section.get("items", [])
            if not items:
                break
            insurer = quote.get("insurer", {}).get("name", "this plan")
            plan_name = quote.get("plan", {}).get("name", "")
            lines = [f"Add-ons for {insurer} ({plan_name}):"]
            for item in items:
                label = item.get("label", "Add-on")
                amount = item.get("amount")
                currency = item.get("currency", "")
                value = item.get("value")
                price_str = f"+{_fmt_amount(amount)} {currency}" if amount is not None else "price on request"
                covers_str = f", covers up to {value:,.0f} {currency}" if isinstance(value, (int, float)) and value else ""
                lines.append(f"- {label}: {price_str}{covers_str}")
            return "\n".join(lines)
    return "This plan doesn't list any optional add-ons."


# A curated subset of coverage line items to show in a compare table — full
# section dumps would be unreadable in chat, so this mirrors the kind of
# curated view the website's own "Compare Quotes" panel shows.
_COMPARE_LABELS = [
    "Emergency Medical Expenses", "Personal Liability", "Delayed Baggage",
    "Missed Departure", "Personal Baggage", "Repatriation of mortal remains",
    "Terrorism extension",
]


def _find_section_item(quote: dict, label: str):
    for section in quote.get("sections", []):
        for item in section.get("items", []):
            if item.get("label", "").strip().lower() == label.lower():
                return item
    return None


def format_compare_quotes(quotes: list, indices: list, departure: str = "") -> tuple:
    """Returns (intro_text, table). intro_text ("Comparing X and Y:") is the
    only text streamed into the chat bubble; table is
    {"columns": [name, ...], "rows": [{"label": ..., "values": [...]}]},
    attached separately via ChatResponse.ui (ui.type == "comparison_table")
    for the frontend to render as a real HTML <table> — the chat bubble
    itself has no markdown table renderer (confirmed live: a plain-text
    pipe table used to show up as literal '|' and '---' characters), and
    this whole file is Ava-only, so the table renderer this feeds is never
    reachable from Layla's plain RAG replies."""
    selected = [quotes[i] for i in indices if 0 <= i < len(quotes)]
    if len(selected) < 2:
        return "I need at least 2 valid option numbers to compare — try 'compare 1 and 2'.", None

    def col_name(q):
        return f"{q.get('insurer', {}).get('name', 'Insurer')} ({q.get('plan', {}).get('name', '')})"

    names = [col_name(q) for q in selected]
    intro = f"Comparing {_join_natural(names)}:"

    rows = []

    price_values = []
    for q in selected:
        price = _quote_price(q)
        currency = q.get("plan", {}).get("price", {}).get("currency", "")
        if price and currency == "AED":
            price_values.append(format_dual_currency(price, departure))
        elif price:
            price_values.append(f"{_fmt_amount(price)} {currency}")
        else:
            price_values.append("N/A")
    rows.append({"label": "Price", "values": price_values})

    for label in _COMPARE_LABELS:
        item_values, any_found = [], False
        for q in selected:
            item = _find_section_item(q, label)
            if item is None:
                item_values.append("not listed")
                continue
            any_found = True
            val = item.get("value")
            currency = item.get("currency", "")
            if val is False or val is None:
                item_values.append("not covered")
            elif isinstance(val, (int, float)):
                item_values.append(f"{currency} {val:,.0f}".strip())
            else:
                item_values.append(str(val))
        if any_found:
            rows.append({"label": label, "values": item_values})

    return intro, {"columns": names, "rows": rows}


# Rough hemisphere list used only to decide which months count as "winter"
# for a given destination — NEVER used to filter/hide quotes, only to decide
# whether to proactively surface a winter-sports add-on if one exists.
_SOUTHERN_HEMISPHERE_COUNTRIES = {
    "australia", "new zealand", "argentina", "chile", "south africa",
    "brazil", "peru", "uruguay", "bolivia", "namibia", "zimbabwe",
    "fiji", "papua new guinea", "paraguay", "botswana",
}


def _months_in_range(start: datetime, end: datetime) -> set:
    months = set()
    cur = start.replace(day=1)
    while cur <= end:
        months.add(cur.month)
        # jump to the 1st of the next month, safe across month lengths
        cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)
    return months


def trip_overlaps_winter(destination: str, start_date: str, end_date: str) -> bool:
    """Heuristic only — a wrong guess here just means a missed suggestion,
    never incorrect data, since this never filters or hides any quote."""
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return False

    months = _months_in_range(start, end)
    is_southern = (destination or "").strip().lower() in _SOUTHERN_HEMISPHERE_COUNTRIES
    winter_months = {5, 6, 7, 8, 9} if is_southern else {11, 12, 1, 2, 3}
    return bool(months & winter_months)


_WINTER_ADDON_KEYWORDS = ["winter sport", "ski", "snow"]


def find_addon_by_keyword(quote: dict, keywords: list):
    for section in quote.get("sections", []):
        if section.get("title", "").strip().lower() != "add-ons":
            continue
        for item in section.get("items", []):
            if any(kw in item.get("label", "").lower() for kw in keywords):
                return item
    return None


def build_seasonal_addon_note(state: dict) -> str:
    """Scans the top displayed quotes for a winter-sports-style add-on and
    proactively surfaces it if the trip dates overlap winter for the
    destination's hemisphere. Purely additive — never changes which quotes
    are shown or their order."""
    if not trip_overlaps_winter(state.get("destination", ""), state.get("start_date", ""), state.get("end_date", "")):
        return ""

    hits = []
    for i, q in enumerate(state.get("available_quotes", [])[:5]):
        addon = find_addon_by_keyword(q, _WINTER_ADDON_KEYWORDS)
        if addon:
            insurer = q.get("insurer", {}).get("name", "Insurer")
            amount = addon.get("amount")
            currency = addon.get("currency", "")
            price_str = f"+{_fmt_amount(amount)} {currency}" if amount is not None else "available"
            hits.append(f"- Option {i + 1} ({insurer}): {addon.get('label')} ({price_str})")

    if not hits:
        return ""
    return "\n\n❄️ Since your trip falls in winter, you might want a Winter Sports add-on:\n" + "\n".join(hits)


def _value_score(q: dict):
    """Rough 'coverage per unit price' heuristic using ONE line item
    (Emergency Medical Expenses) — NOT a rigorous comparison. Doesn't
    account for currency differences between the section item (often
    USD/EUR) and the plan price (usually AED), or weigh any other coverage
    dimension. Good enough to break a tie between similarly-priced options,
    not a substitute for the user reading the actual coverage."""
    price = _quote_price(q)
    if not price:
        return None
    med = _find_section_item(q, "Emergency Medical Expenses")
    med_value = med.get("value") if med else None
    if not isinstance(med_value, (int, float)) or med_value <= 0:
        return None
    return med_value / price


def try_fetch_quotes(state: dict, user_message: str = ""):
    """
    If the checklist is complete, fetches live quotes, mutates `state` in place
    (phase + available_quotes), and returns the reply text. Returns None if the
    checklist isn't complete yet (caller should fall back to asking for more info).
    """
    if not is_quote_checklist_complete(state):
        return None

    # Marketing consent isn't part of the trip checklist — it's only needed
    # for Protego's create-session call — so it's handled as its own short
    # gate right here, asked once before we ever talk to Protego.
    if "marketing_consent" not in state:
        if state.get("_awaiting_marketing_consent"):
            answer = parse_yes_no(user_message)
            if not answer:
                return MARKETING_CONSENT_QUESTION
            state["marketing_consent"] = answer
            state["_awaiting_marketing_consent"] = False
        else:
            state["_awaiting_marketing_consent"] = True
            return MARKETING_CONSENT_QUESTION

    # get-quotes/bind-quotes/issue-policy all reference a session_id AND a
    # client_id that must come from Protego's own create-session endpoint —
    # inventing them (the old random.randint / hardcoded "1618" approach) is
    # what caused "Session not found" 404s. Obtain both once per chat session
    # and reuse them from then on.
    if not state.get("protego_session_id") or not state.get("protego_client_id"):
        session_info = QuoteService.create_session(state)
        if not session_info:
            return "I couldn't start a quoting session with our insurance partner just now. Please try again in a moment!"
        state["protego_session_id"] = session_info["session_id"]
        state["protego_client_id"] = session_info["client_id"]

    raw_quotes = QuoteService.fetch_live_quotes(state)
    if raw_quotes:
        state["available_quotes"] = raw_quotes
        state["phase"] = "CHOOSING"

        # The per-quote data from Protego is just insurer/plan/price — it
        # doesn't echo back the trip dates or destination even though we sent
        # them in the request payload. Show them here so the user can confirm
        # what they're actually being quoted for.
        traveller_count = total_travellers(state)
        traveller_note = f" ({traveller_count} travellers)" if state.get("cover_type") in MULTI_TRAVELLER_COVER_TYPES else ""
        trip_summary = (
            f"Trip: {state.get('destination', '')} · "
            f"{_format_date_human(state.get('start_date', ''))} → {_format_date_human(state.get('end_date', ''))} · "
            f"{state.get('plan_type', '')} · {state.get('cover_type', '')} cover{traveller_note}\n\n"
        )

        reply = trip_summary + "🎉 Great news! I have all your details and managed to fetch your live options:\n\n"
        reply += format_quotes_list(state, sort_by="price_asc")
        reply += build_seasonal_addon_note(state)
        return reply

    return "I pulled your travel configuration, but the live quoting service is responding blank. Try asking again in a moment!"


def get_or_create_session(db: DBSession, session_id: str) -> ChatSession:
    db_session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
    if not db_session:
        db_session = ChatSession(
            session_id=session_id,
            extracted_data={"phase": "QUOTING"},
            required_fields=REQUIRED_FIELDS,
            obtained_fields={},
            missing_fields=REQUIRED_FIELDS,
        )
        db.add(db_session)
        db.commit()

    # Parallel Request row (request_id == session_id here) per the generic
    # Field Definitions / Request / Extracted Values design — this is what
    # any future agent/handover/other-request-type consumer would query,
    # rather than reaching into ChatSession.extracted_data directly.
    request_row = db.query(Request).filter(Request.request_id == session_id).first()
    if not request_row:
        db.add(Request(request_id=session_id, request_type="Travel Insurance", status="IN_PROGRESS"))
        db.commit()

    return db_session


@router.post("/", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, db: DBSession = Depends(get_db)):
    try:
        db_session = get_or_create_session(db, request.session_id)

        state = db_session.extracted_data or {}
        current_phase = state.get("phase", "QUOTING")

        db.add(Message(session_id=request.session_id, role="user", content=request.message))
        db.commit()

        # A question about the process/quote itself ("what is coverage
        # type?", "how do I fill this in?") — answered directly by Ava's
        # own LLM and returned immediately, without touching state/phase at
        # all, so the in-progress form/flow resumes exactly where it left
        # off on the next message. Checked before the phase machine so it
        # applies in every phase (QUOTING/CHOOSING/BINDING/ISSUING) — it's a
        # pure read with zero side effects, safe everywhere.
        if is_ava_process_question(request.message):
            answer = answer_ava_process_question(request.message, state)
            db.add(Message(session_id=request.session_id, role="assistant", content=answer))
            db.commit()
            return ChatResponse(response=answer, options=determine_reply_options(state, answer) or None)

        final_reply = ""
        comparison_table = None  # set only by the CHOOSING phase's "compare" branch below
        showed_quote_list = False  # set wherever a quote list is actually shown, for the "email me this quote" button

        if current_phase == "QUOTING":
            # 1. Extract/update fields from the latest message (flat, single-message schema —
            #    see extract_fields_from_conversation for why nested/history-based schemas were removed)
            previously_missing = {f for f in REQUIRED_FIELDS if not state.get(f)}

            # A bare "Hi"/"Hey there!" can't contain any extractable field —
            # skip the LLM round-trip entirely rather than paying for one
            # just to get an empty result back every time.
            if is_greeting_only(request.message):
                updated_state = {}
            else:
                updated_state = llm_service.extract_fields_from_conversation(db, request.session_id, request.message)
            state = merge_extracted_fields(state, updated_state)
            apply_coverage_type_locks(state)

            newly_filled = [f for f in REQUIRED_FIELDS if f in previously_missing and state.get(f)]

            for field in newly_filled:
                db.add(FieldExtractionLog(
                    session_id=request.session_id,
                    field_name=field,
                    field_value=str(state.get(field, "")),
                    source_message=request.message,
                ))
            sync_extracted_values(db, request.session_id, state, newly_filled)

            print(f"\n--- CURRENT MEMORY STATE ---\n{state}\n----------------------------\n")

            # 2. Catch coverage_type / destination / departure mismatches (e.g.
            #    "GCC Countries" coverage with "France" as the destination)
            #    before treating the checklist as complete.
            bad_field, consistency_error = check_coverage_consistency(state)
            if consistency_error:
                state.pop(bad_field, None)
                final_reply = consistency_error
            elif not is_quote_checklist_complete(state):
                final_reply = build_missing_fields_reply(state, newly_filled=newly_filled, user_message=request.message)
            elif state.get("cover_type") in MULTI_TRAVELLER_COVER_TYPES:
                # 3. Group/Family policies need a full traveller list (2-5 people).
                #    Try to pull a companion out of THIS message (harmless if it's
                #    just "done" or unrelated — extraction returns empty fields).
                if total_travellers(state) < MAX_TRAVELLERS_TOTAL:
                    companion = llm_service.extract_companion_traveller(request.message)
                    if companion.get("first_name") and companion.get("date_of_birth"):
                        state.setdefault("additional_travellers", [])
                        state["additional_travellers"].append(companion)
                        db.add(FieldExtractionLog(
                            session_id=request.session_id,
                            field_name="additional_traveller",
                            field_value=f"{companion.get('first_name', '')} {companion.get('last_name', '')} (DOB: {companion.get('date_of_birth', '')})".strip(),
                            source_message=request.message,
                        ))
                        # "Traveler Details" (list) + "Number of Travelers" per
                        # the field definitions — upserted as the current full
                        # list/count, not appended, since these are single
                        # current-value fields, unlike the log line above.
                        all_travellers = [{
                            "first_name": state.get("first_name", ""),
                            "last_name": state.get("last_name", ""),
                            "date_of_birth": state.get("date_of_birth", ""),
                        }] + state["additional_travellers"]
                        upsert_extracted_value(db, request.session_id, "traveler_details", json.dumps(all_travellers))
                        upsert_extracted_value(db, request.session_id, "number_of_travelers", str(len(all_travellers)))

                if not travellers_requirement_met(state):
                    final_reply = build_traveller_prompt(state, just_selected=("cover_type" in newly_filled))
                elif total_travellers(state) < MAX_TRAVELLERS_TOTAL and not is_done_signal(request.message):
                    final_reply = build_traveller_prompt(state)
                else:
                    quote_reply = try_fetch_quotes(state, request.message)
                    final_reply = quote_reply if quote_reply is not None else build_missing_fields_reply(state, newly_filled=newly_filled, user_message=request.message)
                    showed_quote_list = quote_reply is not None and state.get("phase") == "CHOOSING"
            else:
                # 4. Individual policy with a complete checklist — fetch quotes now.
                quote_reply = try_fetch_quotes(state, request.message)
                final_reply = quote_reply if quote_reply is not None else build_missing_fields_reply(state, newly_filled=newly_filled, user_message=request.message)
                showed_quote_list = quote_reply is not None and state.get("phase") == "CHOOSING"

        elif current_phase == "CHOOSING":
            msg_clean = request.message.strip()
            msg_lower = msg_clean.lower()
            quotes = state.get("available_quotes", [])

            sort_by = parse_sort_command(msg_lower)
            wants_addons = any(kw in msg_lower for kw in ["add-on", "addon", "add on"])
            wants_compare = "compare" in msg_lower

            if sort_by:
                final_reply = format_quotes_list(state, sort_by=sort_by)
                showed_quote_list = True
            elif wants_compare:
                indices = parse_compare_indices(msg_lower)
                final_reply, comparison_table = format_compare_quotes(quotes, indices, departure=state.get("departure", ""))
            elif wants_addons:
                idx = parse_option_number(msg_lower)
                if 0 <= idx < len(quotes):
                    final_reply = format_addons_for_quote(quotes[idx])
                else:
                    final_reply = f"Which option's add-ons would you like? I have {len(quotes)} listed above."
            elif msg_clean in [str(i) for i in range(1, 6)]:
                choice_idx = int(msg_clean) - 1
                if choice_idx < len(quotes):
                    state["selected_quote_index"] = choice_idx
                    state["phase"] = "BINDING"
                    final_reply = "Perfect! To bind this quote to your official profile, please type your Passport Number."
                else:
                    final_reply = f"I only have {len(quotes)} option(s) listed above — please pick one of those numbers."
            else:
                final_reply = (
                    "Please reply with a number to select a plan, or ask me to sort by price/insurer, "
                    "show add-ons for option N, or compare 1 and 2 (etc.)."
                )

        elif current_phase == "BINDING":
            if "passport_number" not in state:
                state["passport_number"] = request.message.strip().upper()
                final_reply = "Got it. Now please provide your Emirates ID (or local identification number)."
            elif "emirates_id" not in state:
                state["emirates_id"] = request.message.strip()

                bind_response = QuoteService.bind_quote(state, state["selected_quote_index"])
                booking_ref = bind_response.get("booking_reference_id")

                if booking_ref:
                    # Keep anything else Protego returned (insurer_policy_number,
                    # provider_transaction_id, etc.) — issue_policy needs it later.
                    state.update(bind_response)
                    state["booking_reference_id"] = booking_ref
                    state["phase"] = "ISSUING"
                    final_reply = (
                        f"🔒 Quote Bound Successfully!\n"
                        f"Your temporary Booking Reference is: {booking_ref}.\n\n"
                        f"Type 'issue' whenever you are ready to authorize checkout and deploy coverage."
                    )
                else:
                    final_reply = "Validation failed on binding parameters. Let's try confirming your passport number again."
                    state.pop("passport_number", None)

        elif current_phase == "ISSUING":
            if "issue" in request.message.lower():
                issue_response = QuoteService.issue_policy(state)
                policy_number = issue_response.get("policy_number", "POL-DEPL-9982")

                state["phase"] = "COMPLETED"
                state["policy_number"] = policy_number

                final_reply = (
                    f"✈️ Transaction Complete! Your Trip is Insured!\n\n"
                    f"Your active policy number is {policy_number}. The certificate has been directly transmitted to your verified email."
                )
            else:
                final_reply = "Your order is pending confirmation. Type 'issue' to finalize deployment."

        else:
            final_reply = "Your policy assignment workflow has been fully executed! Let me know if you need to plan another trip."

        db_session.extracted_data = state
        flag_modified(db_session, "extracted_data")
        sync_field_tracking(db_session, state)

        db.add(Message(session_id=request.session_id, role="assistant", content=final_reply))
        db.commit()

        if comparison_table:
            ui = {"type": "comparison_table", "table": comparison_table}
        elif showed_quote_list:
            ui = {"type": "quote_list"}
        else:
            ui = None
        return ChatResponse(response=final_reply, options=determine_reply_options(state, final_reply) or None, ui=ui)

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-document", response_model=ChatResponse)
def upload_document_endpoint(
    session_id: str = Form(...),
    file: UploadFile = File(...),
    db: DBSession = Depends(get_db),
):
    """
    Handles passport/ticket/visa uploads. Flow:
      1. Send the file to Protego's /extract-pdf.
      2. Normalize whatever it returns into our canonical field names
         (deterministic, history-free — never touches the chat transcript).
      3. Merge into session state.
      4. If the checklist is now complete, fetch live quotes immediately.
         Otherwise tell the user what was captured and what's still missing.
    """
    try:
        db_session = get_or_create_session(db, session_id)
        state = db_session.extracted_data or {}
        state.setdefault("phase", "QUOTING")
        showed_quote_list = False

        db.add(Message(session_id=session_id, role="user", content=f"📎 Uploaded document: {file.filename}"))
        db.commit()

        file_bytes = file.file.read()
        raw_extraction = QuoteService.extract_pdf_bytes(file_bytes, file.filename, file.content_type)
        print(f"[upload-document] raw Protego extract-pdf response: {raw_extraction}")

        if not raw_extraction:
            final_reply = "I couldn't read that document. Could you try a clearer scan/photo, or just tell me your details directly?"
        else:
            normalized_fields = llm_service.parse_extracted_fields(raw_extraction)
            state = merge_extracted_fields(state, normalized_fields)

            doc_fields = [f for f in REQUIRED_FIELDS if normalized_fields.get(f)]
            for field in doc_fields:
                db.add(FieldExtractionLog(
                    session_id=session_id,
                    field_name=field,
                    field_value=str(state.get(field, "")),
                    source_message=f"📎 Uploaded document: {file.filename}",
                ))
            sync_extracted_values(db, session_id, state, doc_fields)

            # Summarize only what THIS document actually contributed — not the
            # whole session state, so the message reads as "here's what I got
            # from your upload" rather than re-listing everything ever known.
            obtained_summary = build_obtained_summary(normalized_fields)
            preamble = (obtained_summary + " ") if obtained_summary else ""

            # A flight ticket/passport gives us a destination but never states
            # a coverage_type explicitly — infer it where it's unambiguous.
            if not state.get("coverage_type") and state.get("destination"):
                inferred = infer_coverage_type_from_destination(state["destination"])
                if inferred:
                    state["coverage_type"] = inferred
            apply_coverage_type_locks(state)

            bad_field, consistency_error = check_coverage_consistency(state)
            if consistency_error:
                state.pop(bad_field, None)
                final_reply = consistency_error
            elif (
                state.get("phase") == "QUOTING"
                and is_quote_checklist_complete(state)
                and not travellers_requirement_met(state)
            ):
                # Document extraction only ever gives us the primary traveller —
                # if cover_type turned out to be Group/Family, we still need
                # companion names/DOBs before quotes can be fetched.
                final_reply = preamble + build_traveller_prompt(state)
            else:
                quote_reply = try_fetch_quotes(state) if state.get("phase") == "QUOTING" else None
                if quote_reply is not None:
                    final_reply = quote_reply
                    showed_quote_list = state.get("phase") == "CHOOSING"
                else:
                    final_reply = build_missing_fields_reply(state, preamble=preamble)

        db_session.extracted_data = state
        flag_modified(db_session, "extracted_data")
        sync_field_tracking(db_session, state)

        db.add(Message(session_id=session_id, role="assistant", content=final_reply))
        db.commit()

        return ChatResponse(
            response=final_reply,
            options=determine_reply_options(state, final_reply) or None,
            collected_fields=state,
            ui={"type": "quote_list"} if showed_quote_list else None,
        )

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit-intake-form", response_model=ChatResponse)
def submit_intake_form_endpoint(payload: IntakeFormRequest, db: DBSession = Depends(get_db)):
    """
    Structured, single-shot alternative to the QUOTING-phase conversational
    flow above: the frontend's intake form collects all REQUIRED_FIELDS
    directly (typed inputs, not free text), so no LLM extraction is
    needed — values are written straight into state. Deliberately reuses
    the exact same validation and quote-fetch functions the conversational
    flow uses (check_coverage_consistency, travellers_requirement_met,
    try_fetch_quotes) so a form submission and an equivalent conversational
    answer produce identical downstream behavior — this endpoint only
    changes HOW the fields are collected, not what happens once they are.
    """
    try:
        db_session = get_or_create_session(db, payload.session_id)
        state = db_session.extracted_data or {}
        state.setdefault("phase", "QUOTING")

        if state.get("phase") not in ("QUOTING", "CHOOSING"):
            raise HTTPException(
                status_code=409,
                detail="You've already selected a quote for this trip — details can no longer be edited.",
            )

        # CHOOSING means a quote was already fetched once — this call is an
        # edit (via the summary card's "Edit details" button), not a first
        # submission. Tracked so the protego-session staleness check below
        # only applies to edits, and so the logged user message reads right.
        was_editing = state.get("phase") == "CHOOSING"
        prior_contact = {k: state.get(k, "") for k in ("first_name", "last_name", "email", "mobile_number")}

        db.add(Message(
            session_id=payload.session_id, role="user",
            content="✏️ Updated intake form" if was_editing else "📝 Submitted intake form",
        ))
        db.commit()

        field_values = {
            "first_name": payload.first_name.strip(),
            "last_name": payload.last_name.strip(),
            "email": payload.email.strip(),
            "mobile_number": payload.mobile_number.strip(),
            "coverage_type": payload.coverage_type.strip(),
            "destination": normalize_country(payload.destination.strip()) if payload.destination.strip() else "",
            "departure": normalize_country(payload.departure.strip()) if payload.departure.strip() else "",
            "start_date": normalize_date(payload.start_date.strip()) if payload.start_date.strip() else "",
            "end_date": normalize_date(payload.end_date.strip()) if payload.end_date.strip() else "",
            "plan_type": payload.plan_type.strip(),
            "cover_type": payload.cover_type.strip(),
            "date_of_birth": normalize_date(payload.date_of_birth.strip()) if payload.date_of_birth.strip() else "",
        }

        # Backstop for a coverage_type that locks destination/departure to a
        # single valid value (see apply_coverage_type_locks) — the frontend
        # is expected to already pre-fill/disable these before the user can
        # even submit, but a direct API call or a stale client shouldn't be
        # forced to guess the one correct value either.
        locked = COVERAGE_TYPE_LOCKS.get(field_values.get("coverage_type", ""), {})
        for field, value in locked.items():
            if not field_values.get(field):
                field_values[field] = value

        missing = [f for f in REQUIRED_FIELDS if not field_values.get(f)]
        if missing:
            raise HTTPException(
                status_code=422,
                detail={"error": "missing_fields", "fields": missing},
            )

        # Backstop for the same two checks the form already does inline
        # (isValidEmail/isValidPhoneForCountry in app.js) — a direct API
        # call or a stale client shouldn't be able to skip past either.
        # Deliberately permissive on email (matches the frontend's own
        # regex — accepts business/personal/education addresses alike, not
        # a strict RFC 5322 pattern that rejects valid real-world ones) and
        # generic on phone (E.164 shape only — the frontend is where the
        # richer per-country length check lives, since that table only
        # needs to exist once).
        if not _EMAIL_RE.match(field_values["email"]):
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_field", "field": "email", "message": "Please enter a valid email address."},
            )
        if not re.match(r"^\+?[1-9]\d{6,14}$", field_values["mobile_number"]):
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_field", "field": "mobile_number", "message": "Please enter a valid mobile number."},
            )

        # Backstop for the same calendar bounds the form's date inputs
        # already enforce via min/max (date_of_birth can't be in the
        # future; a trip can't start in the past or end before it starts).
        # normalize_date already converted these to YYYY-MM-DD, which
        # compares correctly as plain strings.
        today_str = datetime.now().strftime("%Y-%m-%d")
        if re.match(r"^\d{4}-\d{2}-\d{2}$", field_values["date_of_birth"]) and field_values["date_of_birth"] > today_str:
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_field", "field": "date_of_birth", "message": "Date of birth can't be in the future."},
            )
        if re.match(r"^\d{4}-\d{2}-\d{2}$", field_values["start_date"]) and field_values["start_date"] < today_str:
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_field", "field": "start_date", "message": "Trip start date can't be in the past."},
            )
        if (
            re.match(r"^\d{4}-\d{2}-\d{2}$", field_values["end_date"])
            and re.match(r"^\d{4}-\d{2}-\d{2}$", field_values["start_date"])
            and field_values["end_date"] < field_values["start_date"]
        ):
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_field", "field": "end_date", "message": "Trip end date can't be before the start date."},
            )

        state.update(field_values)

        # create_session's payload is built from exactly these four fields
        # and its result (protego_session_id/client_id) is cached forever
        # once set — editing one of them and re-fetching without clearing
        # the cache would leave Protego's own session record tied to the
        # pre-edit identity while the rest of the quote request uses the
        # corrected values. fetch_live_quotes itself is fully stateless, so
        # this is the only staleness risk a second submission introduces.
        if was_editing and any(state.get(k, "") != prior_contact[k] for k in prior_contact):
            state.pop("protego_session_id", None)
            state.pop("protego_client_id", None)

        # Same consistency check the conversational flow runs (chat_endpoint,
        # QUOTING branch) — pops the offending field on conflict so the form
        # re-asks for just that one, not the whole form.
        bad_field, consistency_error = check_coverage_consistency(state)
        if consistency_error:
            state.pop(bad_field, None)
            db_session.extracted_data = state
            flag_modified(db_session, "extracted_data")
            sync_field_tracking(db_session, state)
            db.commit()
            raise HTTPException(
                status_code=422,
                detail={"error": "field_conflict", "field": bad_field, "message": consistency_error},
            )

        if state.get("cover_type") in MULTI_TRAVELLER_COVER_TYPES:
            travellers = [
                {
                    "first_name": t.first_name.strip(),
                    "last_name": t.last_name.strip(),
                    "date_of_birth": normalize_date(t.date_of_birth.strip()) if t.date_of_birth.strip() else "",
                }
                for t in payload.additional_travellers
                if t.first_name.strip() and t.date_of_birth.strip()
            ]
            if 1 + len(travellers) > MAX_TRAVELLERS_TOTAL:
                travellers = travellers[: MAX_TRAVELLERS_TOTAL - 1]
            state["additional_travellers"] = travellers

            if not travellers_requirement_met(state):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "travellers_required",
                        "message": f"{state['cover_type']} cover needs at least "
                                   f"{MIN_TRAVELLERS_FOR_MULTI} travellers — add at least "
                                   f"{MIN_TRAVELLERS_FOR_MULTI - 1} companion(s) with a name and date of birth.",
                    },
                )

            all_travellers = [{
                "first_name": state.get("first_name", ""),
                "last_name": state.get("last_name", ""),
                "date_of_birth": state.get("date_of_birth", ""),
            }] + travellers
            upsert_extracted_value(db, payload.session_id, "traveler_details", json.dumps(all_travellers))
            upsert_extracted_value(db, payload.session_id, "number_of_travelers", str(len(all_travellers)))

        for field in REQUIRED_FIELDS:
            db.add(FieldExtractionLog(
                session_id=payload.session_id,
                field_name=field,
                field_value=str(state.get(field, "")),
                source_message="📝 Submitted intake form",
            ))
        sync_extracted_values(db, payload.session_id, state, REQUIRED_FIELDS)

        # Set directly from the form's checkbox — try_fetch_quotes's own
        # marketing-consent gate only fires when the key is absent, so this
        # skips that conversational yes/no round-trip entirely.
        state["marketing_consent"] = "yes" if payload.marketing_consent else "no"

        quote_reply = try_fetch_quotes(state)
        final_reply = quote_reply if quote_reply is not None else build_missing_fields_reply(state)
        showed_quote_list = quote_reply is not None and state.get("phase") == "CHOOSING"

        db_session.extracted_data = state
        flag_modified(db_session, "extracted_data")
        sync_field_tracking(db_session, state)

        db.add(Message(session_id=payload.session_id, role="assistant", content=final_reply))
        db.commit()

        ui = {"type": "quote_list"} if showed_quote_list else None
        return ChatResponse(response=final_reply, collected_fields=state, ui=ui)

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/email-quote", response_model=ChatResponse)
def email_quote_endpoint(payload: EmailQuoteRequest, db: DBSession = Depends(get_db)):
    """Sends the customer their currently-available quotes by email. Reads
    quote data fresh from the session's own state (never trusts anything
    the client might send about which quotes are shown) — the "Email me
    this quote" button just tells this endpoint who to email, nothing
    about what's in the email itself."""
    try:
        db_session = get_or_create_session(db, payload.session_id)
        state = db_session.extracted_data or {}

        quotes = state.get("available_quotes") or []
        if not quotes:
            raise HTTPException(
                status_code=422,
                detail={"error": "no_quotes", "message": "There's no quote to email yet — get a quote first."},
            )

        email = payload.email.strip()
        if not _EMAIL_RE.match(email):
            raise HTTPException(
                status_code=422,
                detail={"error": "invalid_field", "field": "email", "message": "Please enter a valid email address."},
            )

        from email_utils import compose_quotation_email_body, send_quotation_email

        recipient_name = state.get("first_name", "").strip()
        trip_summary = trip_summary_for_email(state)
        quotes_text = quotes_summary_for_email(state)
        body_html = compose_quotation_email_body(payload.session_id, recipient_name, trip_summary, quotes_text)

        sent = send_quotation_email(email, "Your InsureHub Travel Insurance Quote", body_html)
        if not sent:
            raise HTTPException(
                status_code=502,
                detail="Couldn't send the email right now — please try again in a moment.",
            )

        final_reply = f"✅ Sent your quote to {email}."
        db.add(Message(session_id=payload.session_id, role="user", content=f"📧 Requested quote by email to {email}"))
        db.add(Message(session_id=payload.session_id, role="assistant", content=final_reply))
        db.commit()

        return ChatResponse(response=final_reply)

    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))