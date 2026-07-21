import traceback
from sqlalchemy.orm import Session
from travel_bot.models.session import ChatSession
from travel_bot.schemas.travel import TravelInsuranceDetails, CompanionTraveller


class LLMService:
    """
    Uses the SAME LLM backend as the rest of the app (Layla's RAG side) —
    routed through router.get_insurance_llm(), which respects whatever
    backend Super Admin has selected (Auto/vLLM/Groq/Manual/OpenAI/
    Anthropic). No separate API key or provider-specific client here
    anymore — this used to call OpenAI directly with its own key, which
    is no longer safe to do (that key was leaked to a public GitHub repo
    and is considered compromised).

    Structured extraction now goes through LangChain's provider-agnostic
    with_structured_output() instead of OpenAI-SDK-specific
    beta.chat.completions.parse(). Reliability of structured output
    depends on whatever backend is currently active — if Super Admin has
    selected vLLM, that server needs guided-decoding support configured
    for consistent results; if extraction quality degrades, check the
    active backend in Super Admin before assuming a code regression here.
    """

    def _get_llm(self, temperature: float = 0.0):
        from router import get_insurance_llm
        return get_insurance_llm(temperature=temperature)

    def extract_companion_traveller(self, user_message: str) -> dict:
        """
        Extracts ONE additional traveller's name + DOB from a message, for
        Group/Family policies that need more than one person on the policy.
        Returns {} fields as empty strings if this message doesn't mention
        a companion at all.
        """
        prompt = (
            "The user is adding an ADDITIONAL traveller to a family/group travel "
            "insurance policy (not themselves — they're already registered as the "
            "primary traveller). Extract this companion's first name, last name, "
            "and date of birth from the message below, if present.\n"
            "Normalize the date of birth to YYYY-MM-DD.\n"
            "If this message doesn't name a new traveller at all (e.g. it just says "
            "'done' or asks a question), leave every field as an empty string — "
            "never guess a name.\n\n"
            f"USER MESSAGE:\n{user_message}"
        )
        try:
            llm = self._get_llm(temperature=0.0)
            structured_llm = llm.with_structured_output(CompanionTraveller)
            parsed = structured_llm.invoke(prompt)
            extracted = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
            print(f"[extract_companion_traveller] extracted: {extracted}")
            return extracted
        except Exception:
            print("[extract_companion_traveller] LLM extraction failed:")
            print(traceback.format_exc())
            return {}

    def extract_fields_from_conversation(self, db: Session, session_id: str, user_message: str) -> dict:
        """
        Extracts travel-insurance fields from the user's LATEST message.
        Single-message pattern — merge_extracted_fields in chat.py progressively
        builds up state turn-by-turn. Returns the extracted fields dict
        (possibly empty — callers should treat that as "no new info this
        turn", not a hard error).
        """
        db_session = db.query(ChatSession).filter(ChatSession.session_id == session_id).first()
        state = db_session.extracted_data or {}

        required_fields = [
            "first_name", "last_name", "email", "mobile_number",
            "coverage_type", "destination", "start_date", "end_date",
            "plan_type", "cover_type", "date_of_birth"
        ]
        missing_fields = [f for f in required_fields if not state.get(f)]

        if not missing_fields:
            return {}

        prompt = (
            "Extract travel-insurance details from the user's message below. "
            f"Focus especially on these fields, which are still missing: {missing_fields}. "
            "Only fill a field if you can confidently determine it from this message. "
            "Leave anything uncertain or not mentioned as an empty string — never guess.\n\n"
            "EXTRACTION RULES (STRICT) — three fields are easy to confuse, keep them separate:\n"
            "  - coverage_type: the INSURANCE PRODUCT. Must be exactly one of: 'Hajj and Umrah', "
            "'UAE Inbound', 'Worldwide', 'Schengen', 'GCC Countries'. Infer it from where they're going — "
            "e.g. 'going to Portugal' -> 'Schengen'; 'flying to Dubai' -> 'UAE Inbound'; "
            "'going to Qatar' -> 'GCC Countries'; 'performing Umrah' -> 'Hajj and Umrah'; otherwise -> 'Worldwide'.\n"
            "  - destination: the actual destination COUNTRY named (e.g. 'Portugal', 'Qatar'). "
            "Never put the coverage_type value here.\n"
            "  - cover_type: WHO is insured — exactly one of 'Individual', 'Group', or 'Family'. "
            "Has nothing to do with coverage_type or destination.\n"
            "  - plan_type: exactly 'Single Trip' or 'Annual Multi-Trip' (map 'single' -> 'Single Trip', "
            "'annual'/'multi-trip' -> 'Annual Multi-Trip').\n"
            "  - first_name / last_name: split full names intelligently.\n"
            "  - date_of_birth: normalize to YYYY-MM-DD (e.g. '21/09/2001' -> '2001-09-21').\n\n"
            f"USER MESSAGE:\n{user_message}"
        )

        try:
            llm = self._get_llm(temperature=0.0)
            structured_llm = llm.with_structured_output(TravelInsuranceDetails)
            parsed = structured_llm.invoke(prompt)
            extracted = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
            print(f"[extract_fields_from_conversation] extracted: {extracted}")
            return extracted
        except Exception:
            print("[extract_fields_from_conversation] LLM extraction failed:")
            print(traceback.format_exc())
            return {}

    # Fields we can plausibly get out of a travel document. plan_type / cover_type
    # are deliberately excluded — a flight ticket or passport has no concept of
    # insurance plan/cover type, so those always require manual entry.
    _EXTRACTABLE_FIELDS = [
        "first_name", "last_name", "email", "mobile_number",
        "destination", "start_date", "end_date", "date_of_birth"
    ]

    def _deterministic_field_map(self, raw_extraction: dict) -> dict:
        """
        Maps Protego's known /extract-pdf response shape directly, with no LLM
        call involved.
        """
        if not isinstance(raw_extraction, dict):
            return {}

        results = raw_extraction.get("results")
        if isinstance(results, list) and results:
            sd = results[0].get("structured_data", {}) or {}
        elif isinstance(raw_extraction.get("structured_data"), dict):
            sd = raw_extraction["structured_data"]
        else:
            sd = raw_extraction

        norm = {}
        for k, v in (sd or {}).items():
            if v in (None, ""):
                continue
            key = str(k).lower().replace(" ", "").replace("_", "")
            norm[key] = v

        mapped = {}

        full_name = norm.get("insuredname") or norm.get("name") or norm.get("passengername")
        if full_name:
            parts = str(full_name).strip().split(" ", 1)
            mapped["first_name"] = parts[0]
            if len(parts) > 1:
                mapped["last_name"] = parts[1]

        if norm.get("email"):
            mapped["email"] = norm["email"]

        phone = norm.get("phonenumber") or norm.get("mobilenumber") or norm.get("contactnumber")
        if phone:
            mapped["mobile_number"] = str(phone).replace(" ", "")

        destination = norm.get("destinationcountry") or norm.get("destination")
        if destination:
            mapped["destination"] = destination

        start_date = norm.get("fromdate") or norm.get("departuredate") or norm.get("startdate")
        if start_date:
            mapped["start_date"] = start_date

        end_date = norm.get("todate") or norm.get("returndate") or norm.get("enddate")
        if end_date:
            mapped["end_date"] = end_date

        dob = norm.get("dateofbirth") or norm.get("dob")
        if dob:
            mapped["date_of_birth"] = dob

        return mapped

    def parse_extracted_fields(self, raw_extraction: dict) -> dict:
        """
        Normalizes the JSON returned by the document-extraction engine (Protego's
        /extract-pdf) into our canonical TravelInsuranceDetails field names.

        Two passes:
          1. Deterministic mapping — no LLM involved.
          2. An LLM fallback (via the active backend's structured output) that
             only runs for whatever fields pass 1 didn't find.
        """
        if not raw_extraction:
            print("[parse_extracted_fields] raw_extraction was empty — extract-pdf returned nothing.")
            return {}

        mapped = self._deterministic_field_map(raw_extraction)
        print(f"[parse_extracted_fields] deterministic mapping found: {mapped}")

        still_missing = [f for f in self._EXTRACTABLE_FIELDS if not mapped.get(f)]
        if still_missing:
            import json
            prompt = (
                "Map the following raw document-extraction JSON onto the target schema. "
                f"Focus especially on these fields, which a first pass could not confidently find: {still_missing}. "
                "Only fill a field if you can confidently infer it from the data below. "
                "Leave anything uncertain or absent as an empty string — never guess.\n\n"
                f"RAW EXTRACTION JSON:\n{json.dumps(raw_extraction)}"
            )
            try:
                llm = self._get_llm(temperature=0.0)
                structured_llm = llm.with_structured_output(TravelInsuranceDetails)
                parsed = structured_llm.invoke(prompt)
                llm_fields = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
                print(f"[parse_extracted_fields] LLM fallback found: {llm_fields}")
                for key in still_missing:
                    val = llm_fields.get(key)
                    if val and str(val).strip():
                        mapped[key] = val
            except Exception as e:
                print(f"[parse_extracted_fields] LLM fallback error (deterministic mapping still applies): {e}")

        print(f"[parse_extracted_fields] final merged fields: {mapped}")
        return mapped