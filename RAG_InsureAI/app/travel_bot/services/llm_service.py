import traceback
from datetime import datetime
from sqlalchemy.orm import Session
from travel_bot.models.session import ChatSession
from travel_bot.schemas.travel import TravelInsuranceDetails, CompanionTraveller


class LLMService:
    """
    Prefers Groq directly over router.get_insurance_llm() (Layla's shared
    RAG backend) when GROQ_API_KEY is set — confirmed live 2026-07-22 that
    the shared vLLM host (VLLM_HOST) decodes at only ~7-8 tokens/sec, and
    TravelInsuranceDetails' 12-field schema forces ~90 output tokens on
    every call even when almost every field comes back empty, so a single
    extraction call was taking 10-12s end to end (measured directly against
    vLLM, independent of any LangChain/network overhead). The identical
    call against Groq (llama-3.3-70b-versatile) measured 0.37s — its output
    is the same JSON, just far faster to decode.

    This is safe here even though FORCE_BACKEND=groq stays commented out in
    .env for Layla's RAG side (see the .env comments above GROQ_API_KEY):
    that regression was specifically Groq's generation being unreliable
    against _verify_grounding's YES/NO check on retrieved documents. Travel
    bot has no retrieval/grounding step at all — this is pure structured
    extraction of fields out of the user's own message — so that failure
    mode doesn't apply. Falls back to router.get_insurance_llm() (whatever
    Super Admin has configured) when no Groq key is present, so this still
    works in an environment without one.

    Structured extraction goes through LangChain's provider-agnostic
    with_structured_output() instead of OpenAI-SDK-specific
    beta.chat.completions.parse(). This used to call OpenAI directly with
    its own key, which is no longer safe to do (that key was leaked to a
    public GitHub repo and is considered compromised) — hence going through
    a LangChain chat model instead of a raw provider SDK client here.
    """

    def _get_llm(self, temperature: float = 0.0, max_tokens: int = 0):
        import os
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip(),
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
                temperature=temperature,
                max_tokens=max_tokens if max_tokens > 0 else 500,
                timeout=60,
                max_retries=1,
            )
        from router import get_insurance_llm
        return get_insurance_llm(temperature=temperature, max_tokens=max_tokens)

    def _structured_method(self) -> str:
        """
        with_structured_output()'s method has to match what the active
        backend actually supports — confirmed live these two don't overlap:
        Groq's llama-3.3-70b-versatile rejects response_format=json_schema
        outright (400: "This model does not support response format
        json_schema"), while this deployment's vLLM host rejects tool/
        function-calling (400: "requires --tool-call-parser to be set" —
        it isn't configured). "function_calling" is what's confirmed
        working against Groq; "json_schema" is what was already working
        against vLLM before Groq was ever wired in here.
        """
        import os
        return "function_calling" if os.getenv("GROQ_API_KEY", "").strip() else "json_schema"

    def _structured_kwargs(self) -> dict:
        """Extra with_structured_output() kwargs alongside method(). Groq's
        "function_calling" path defaults to strict=True (LangChain's own
        default), which asks Groq to enforce every schema property present
        with zero tolerance — confirmed live this fails outright and drops
        ALL extracted fields for the turn whenever the model's tool-call
        JSON skips even one property (in practice, always 'departure' —
        TravelInsuranceDetails' one field chat.py never actually asks the
        user for, so the model has nothing to put there and sometimes omits
        the key entirely rather than emitting ""). strict=False lets Groq
        accept a partial-but-valid tool call instead of hard-rejecting it.
        vLLM's "json_schema" path is unaffected either way — its guided
        decoding already structurally forces every key present regardless
        of this flag, which is why the failure never showed up before
        Groq was wired in here.
        """
        return {"strict": False} if self._structured_method() == "function_calling" else {}

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
            # TravelInsuranceDetails/CompanionTraveller are small schemas (a
            # handful of short string fields) — capping output well below
            # vLLM's 1024-token default cuts real generation time for a
            # structured-extraction call that never needs anywhere near that.
            llm = self._get_llm(temperature=0.0, max_tokens=400)
            structured_llm = llm.with_structured_output(CompanionTraveller, method=self._structured_method(), **self._structured_kwargs())
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

        # The model has no other way to know what "today" is — without this,
        # a bare "starting Aug 8, ending Aug 18" (no year given) gets a
        # guessed year from the model's own training-data sense of
        # "current," not the real current year. Confirmed live: this
        # produced a 2024 trip date while running in 2026.
        today = datetime.now().strftime("%Y-%m-%d")

        prompt = (
            f"Today's date is {today}. Extract travel-insurance details from the user's message below. "
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
            "  - start_date / end_date: normalize to YYYY-MM-DD. If the user gives a year, use it exactly. "
            f"If the user does NOT mention a year (e.g. 'August 8th to the 18th'), use today's year ({today[:4]}) "
            "unless that date has already passed this year, in which case use next year instead — a trip is "
            "always in the future, never in the past.\n"
            "  - date_of_birth: normalize to YYYY-MM-DD (e.g. '21/09/2001' -> '2001-09-21'). Unlike trip dates, "
            "NEVER guess a birth year the user didn't give — leave it empty instead.\n\n"
            f"USER MESSAGE:\n{user_message}"
        )

        try:
            # Small schema (a handful of short string fields) — cap output
            # well below vLLM's 1024-token default as a safety ceiling.
            # Confirmed via direct measurement this does NOT fix the real
            # latency here: the model already stops well short of 1024 on
            # its own, so the actual bottleneck is generation speed on the
            # remote vLLM host itself, not an unbounded token ceiling.
            llm = self._get_llm(temperature=0.0, max_tokens=400)
            structured_llm = llm.with_structured_output(TravelInsuranceDetails, method=self._structured_method(), **self._structured_kwargs())
            parsed = structured_llm.invoke(prompt)
            extracted = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
            self._roll_trip_dates_to_future(extracted)
            print(f"[extract_fields_from_conversation] extracted: {extracted}")
            return extracted
        except Exception:
            print("[extract_fields_from_conversation] LLM extraction failed:")
            print(traceback.format_exc())
            return {}

    @staticmethod
    def _roll_trip_dates_to_future(extracted: dict) -> None:
        """Mutates start_date/end_date in place: if start_date has already
        passed, bumps BOTH by the same number of years until start_date is
        in the future. A trip being quoted for insurance is always ahead of
        today, so this is safe to enforce unconditionally on the final
        value regardless of whether the model defaulted the year itself or
        the user actually stated a past one — either way a past start_date
        can only mean "next occurrence of this date," not a real past trip.
        Applied here instead of trusting the prompt's year-inference
        instruction alone: confirmed live the model doesn't reliably reason
        about "has this date passed" through instructions ('January 5' with
        today in July stayed on the current year instead of rolling to next
        year), so the actual date comparison is done in code, not asked of
        the model."""
        start = extracted.get("start_date")
        if not start:
            return
        try:
            start_dt = datetime.strptime(start, "%Y-%m-%d")
            years_ahead = 0
            while start_dt.date() < datetime.now().date():
                years_ahead += 1
                start_dt = start_dt.replace(year=start_dt.year + 1)
        except (ValueError, TypeError):
            # Feb 29 rolling into a non-leap year, or an unparseable value —
            # leave the date exactly as extracted rather than risk losing
            # every other field to the caller's broader except block.
            return
        if years_ahead == 0:
            return
        extracted["start_date"] = start_dt.strftime("%Y-%m-%d")
        end = extracted.get("end_date")
        if end:
            try:
                end_dt = datetime.strptime(end, "%Y-%m-%d")
                extracted["end_date"] = end_dt.replace(year=end_dt.year + years_ahead).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

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
                # Small schema — cap output well below vLLM's 1024-token
                # default (see extract_fields_from_conversation above).
                llm = self._get_llm(temperature=0.0, max_tokens=400)
                structured_llm = llm.with_structured_output(TravelInsuranceDetails, method=self._structured_method(), **self._structured_kwargs())
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