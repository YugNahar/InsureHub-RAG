# app/schemas/travel.py
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Coverage Type -> Destination/Departure rules
#
# IMPORTANT: "coverage_type" (the insurance PRODUCT, e.g. "GCC Countries") is
# a different field from "cover_type" (WHO is insured: Individual/Group/
# Family). These two were previously conflated in the code - this table is
# what actually governs which destinations/departure are valid for each
# coverage type, per the product's own mapping rules.
# ---------------------------------------------------------------------------

GCC_COUNTRIES = ["Bahrain", "Kuwait", "Oman", "Qatar", "Saudi Arabia"]

SCHENGEN_COUNTRIES = [
    "Austria", "Belgium", "Croatia", "Czech Republic", "Denmark", "Estonia",
    "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Italy",
    "Latvia", "Liechtenstein", "Lithuania", "Luxembourg", "Malta",
    "Netherlands", "Norway", "Poland", "Portugal", "Slovakia", "Slovenia",
    "Spain", "Sweden", "Switzerland",
]

COVERAGE_TYPE_RULES = {
    "Hajj and Umrah": {
        "destination_options": ["Saudi Arabia"],
        "departure_options": ["United Arab Emirates"], 
    },
    "UAE Inbound": {
        "destination_options": ["United Arab Emirates"],
        "departure_options": None,  # any country
    },
    "Worldwide": {
        "destination_options": None,  # any country
        "departure_options": ["United Arab Emirates"],
    },
    "Schengen": {
        "destination_options": SCHENGEN_COUNTRIES,
        "departure_options": ["United Arab Emirates"]
    },
    "GCC Countries": {
        "destination_options": GCC_COUNTRIES,
        "departure_options": ["United Arab Emirates"]
    },
}

COVERAGE_TYPE_OPTIONS = list(COVERAGE_TYPE_RULES.keys())

# ---------------------------------------------------------------------------
# Users (and sometimes the LLM extractor) write countries as abbreviations,
# not the full names our validation table checks against ("UAE" vs
# "United Arab Emirates"). This is a closed, known set of countries relevant
# to this product, so a deterministic alias table is the right fix here —
# not an LLM/RAG call. Add to this as new mismatches turn up.
# ---------------------------------------------------------------------------
COUNTRY_ALIASES = {
    "uae": "United Arab Emirates",
    "u.a.e": "United Arab Emirates",
    "u.a.e.": "United Arab Emirates",
    "emirates": "United Arab Emirates",
    "ksa": "Saudi Arabia",
    "k.s.a": "Saudi Arabia",
    "saudi": "Saudi Arabia",
    "uk": "United Kingdom",
    "u.k": "United Kingdom",
    "u.k.": "United Kingdom",
    "usa": "United States",
    "u.s.a": "United States",
    "u.s.a.": "United States",
    "us": "United States",
}


def normalize_country(name: str) -> str:
    """Expands known abbreviations (case/punctuation-insensitive) to the full
    country name our validation/payload logic expects. Unknown values are
    returned unchanged (trimmed) rather than guessed."""
    if not name:
        return name
    key = name.strip().lower().replace(".", "")
    return COUNTRY_ALIASES.get(key, name.strip())


import re
from datetime import datetime

# Protego's API rejects anything that isn't strict ISO (YYYY-MM-DD) — it
# doesn't just prefer it, it 422s on slashes/dots. The extraction prompt asks
# Gemini to normalize dates itself, but that's a request, not a guarantee
# (this is exactly what broke on "15/10/2028" reaching the API verbatim).
# This is a deterministic backstop for the common non-ISO shapes users type.
# Ambiguous MM/DD/YYYY-vs-DD/MM/YYYY input is intentionally NOT auto-handled
# here since this product's user base is UAE-based (day-first is assumed for
# slash/dot formats) — genuinely ambiguous dates should be caught by asking
# the user, not silently guessed.
_DATE_FORMATS_DAYFIRST = ["%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y"]


def normalize_date(value: str) -> str:
    """Converts common day-first date strings to YYYY-MM-DD. Already-ISO or
    unrecognized values are returned unchanged (trimmed) — never guessed."""
    if not value:
        return value
    value = value.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value
    for fmt in _DATE_FORMATS_DAYFIRST:
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value

# ---------------------------------------------------------------------------
# Protego's get-quotes payload expects "coverage_type" as a numeric CODE
# (e.g. "coverage_type": "4"), not the label text like "GCC Countries".
# Confirmed mapping.
COVERAGE_TYPE_CODES = {
    "UAE Inbound": "1",
    "Worldwide": "2",
    "Schengen": "3",
    "GCC Countries": "4",
    "Hajj and Umrah": "5",
}


def get_coverage_type_code(coverage_type: str) -> str:
    """Best-effort mapping to Protego's numeric coverage_type code. See caveat above."""
    return COVERAGE_TYPE_CODES.get(coverage_type, "")


def validate_coverage_destination(coverage_type: str, destination: str, departure: str = "") -> tuple:
    """
    Returns (None, "") if (coverage_type, destination, departure) are
    consistent with the rules above. Otherwise returns (bad_field, message) —
    bad_field names exactly the field that's actually wrong ("coverage_type",
    "destination", or "departure"), so the caller can clear just that field
    instead of guessing. message is a human-readable explanation suitable for
    showing directly to the user.
    """
    rule = COVERAGE_TYPE_RULES.get(coverage_type)
    if not rule:
        return "coverage_type", (
            f"'{coverage_type}' isn't a coverage type I recognize. "
            f"Please choose one of: {', '.join(COVERAGE_TYPE_OPTIONS)}."
        )

    dest_options = rule["destination_options"]
    if destination and dest_options and destination not in dest_options:
        return "destination", (
            f"For '{coverage_type}' coverage, the destination has to be one of: "
            f"{', '.join(dest_options)}. You gave '{destination}' — could you confirm the destination?"
        )

    dep_options = rule["departure_options"]
    if departure and dep_options and departure not in dep_options:
        return "departure", (
            f"For '{coverage_type}' coverage, the trip has to depart from: "
            f"{', '.join(dep_options)}. You gave '{departure}' — could you confirm where you're departing from?"
        )

    return None, ""


def infer_coverage_type_from_destination(destination: str) -> str:
    """
    Best-effort guess of coverage_type from a destination country alone —
    useful right after document extraction, before the user has stated a
    coverage_type explicitly. Returns "" when genuinely ambiguous (e.g.
    'Saudi Arabia' could mean Hajj & Umrah or GCC Countries) rather than
    guessing wrong; the missing-fields prompt will ask the user directly.
    """
    if not destination:
        return ""
    if destination == "Saudi Arabia":
        return ""
    if destination == "United Arab Emirates":
        return "UAE Inbound"
    if destination in GCC_COUNTRIES:
        return "GCC Countries"
    if destination in SCHENGEN_COUNTRIES:
        return "Schengen"
    return "Worldwide"


class TravelInsuranceDetails(BaseModel):
    # default="" on every field: this schema means "fill a field if you can
    # confidently extract it, else leave it empty" (see the prompts in
    # llm_service.py) — none of these are truly required in the business
    # sense, so a partial tool-call response should still validate.
    # Confirmed live 2026-07-22: without a default, the model omitting even
    # one field (in practice, almost always 'departure' — the one field
    # chat.py's REQUIRED_FIELDS never actually asks the user for, so the
    # model usually has nothing to put there and sometimes skips the key
    # entirely instead of emitting "") makes Pydantic raise a hard
    # ValidationError ("Field required"), which discards every OTHER
    # correctly-extracted field for that turn too — a user who just typed
    # their name and email got neither captured, purely because of this
    # one unrelated field. (A prior version of this comment claimed no
    # `default=` was needed because this schema was "only ever used as a
    # schema hint for Gemini calls" — there is no live Gemini backend
    # anywhere in this codebase, router.py has no Gemini branch at all, so
    # that reasoning no longer applies.)
    first_name: str = Field(default="", description="The user's first name")
    last_name: str = Field(default="", description="The user's last name")
    email: str = Field(default="", description="The user's email address")
    mobile_number: str = Field(default="", description="The user's mobile number")

    coverage_type: str = Field(
        default="",
        description=(
            "The INSURANCE PRODUCT the user needs - must be exactly one of: "
            "'Hajj and Umrah', 'UAE Inbound', 'Worldwide', 'Schengen', 'GCC Countries'. "
            "This is NOT the same as cover_type (who is insured) and is NOT the same "
            "as destination (the specific country)."
        )
    )
    destination: str = Field(
        default="",
        description="The specific destination country (e.g. 'Qatar', 'Portugal'). Must be valid for the chosen coverage_type."
    )
    departure: str = Field(
        default="",
        description="The departure/origin country. Must be valid for the chosen coverage_type (most types allow any country)."
    )

    start_date: str = Field(default="", description="Travel start date (YYYY-MM-DD)")
    end_date: str = Field(default="", description="Travel end date (YYYY-MM-DD)")
    plan_type: str = Field(default="", description="Must be 'Single Trip' or 'Annual Multi-Trip'")

    cover_type: str = Field(
        default="",
        description=(
            "WHO is insured - must be exactly one of: 'Individual', 'Group', or 'Family'. "
            "This is NOT the same as coverage_type (the insurance product/destination rules)."
        )
    )
    date_of_birth: str = Field(default="", description="The user's date of birth (YYYY-MM-DD)")


class CompanionTraveller(BaseModel):
    """
    One additional traveller on a Group/Family policy (i.e. NOT the primary
    account holder, who's already captured by TravelInsuranceDetails'
    first_name/last_name/date_of_birth). default="" on every field — see
    TravelInsuranceDetails above for why.
    """
    first_name: str = Field(default="", description="This companion traveller's first name")
    last_name: str = Field(default="", description="This companion traveller's last name")
    date_of_birth: str = Field(default="", description="This companion traveller's date of birth, normalized to YYYY-MM-DD")


# Group/Family policies need a full traveller list (primary + companions).
# Individual is always exactly 1 (the primary traveller only).
MULTI_TRAVELLER_COVER_TYPES = ("Group", "Family")
MIN_TRAVELLERS_FOR_MULTI = 2
MAX_TRAVELLERS_TOTAL = 5