from openai import OpenAI
import json
import traceback
from sqlalchemy.orm import Session
from app.core.config import settings
from app.models.session import ChatSession
from app.schemas.travel import TravelInsuranceDetails, CompanionTraveller

# CONFIRM THIS before running: OpenAI's model naming moves fast, and even cgyḥ...ūḥ
# checking sources dated within the last week disagree on exactly what's GA
# right now vs. still in limited preview. Check
# https://platform.openai.com/docs/models for the current mini/cost-tier
# model name and swap it in below — this is a placeholder, not a verified
# current model id.
MODEL_NAME = "gpt-4o"


class LLMService:
    def __init__(self):
        # Needs OPENAI_API_KEY added to your Settings class (app/core/config.py)
        # and your .env file — this codebase only ever had GEMINI_API_KEY
        # defined, so this will fail until that's added.
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def extract_companion_traveller(self, user_message: str) -> dict:
        """
        Extracts ONE additional traveller's name + DOB from a message, for
        Group/Family policies that need more than one person on the policy.
        Uses OpenAI's Structured Outputs (client.beta.chat.completions.parse
        with response_format=<PydanticModel>) — the direct equivalent of the
        Gemini response_schema= approach this replaced. Returns {} fields as
        empty strings if this message doesn't mention a companion at all.
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
            completion = self.client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format=CompanionTraveller,
                temperature=0.0,
            )
            parsed = completion.choices[0].message.parsed
            extracted = parsed.model_dump() if parsed else {}
            print(f"[extract_companion_traveller] extracted: {extracted}")
            return extracted
        except Exception:
            print("[extract_companion_traveller] LLM extraction failed:")
            print(traceback.format_exc())
            return {}

    def extract_fields_from_conversation(self, db: Session, session_id: str, user_message: str) -> dict:
        """
        Extracts travel-insurance fields from the user's LATEST message.

        Same single-message pattern as the Gemini version this replaced —
        one flat user message, no multi-turn history array, relying on
        merge_extracted_fields in chat.py to progressively build up state
        turn-by-turn. Only the SDK/response_format mechanism changed; the
        prompt and merge strategy did not.

        Returns the extracted fields dict (possibly empty — callers should
        treat that as "no new info this turn", not a hard error).
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
            completion = self.client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                response_format=TravelInsuranceDetails,
                temperature=0.0,
            )
            parsed = completion.choices[0].message.parsed
            extracted = parsed.model_dump() if parsed else {}
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
        call involved. Unchanged from the Gemini version — this logic never
        touched the LLM SDK in the first place.
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
          1. Deterministic mapping — unchanged, no LLM involved.
          2. An LLM fallback (now via OpenAI Structured Outputs) that only runs
             for whatever fields pass 1 didn't find.
        """
        if not raw_extraction:
            print("[parse_extracted_fields] raw_extraction was empty — extract-pdf returned nothing.")
            return {}

        mapped = self._deterministic_field_map(raw_extraction)
        print(f"[parse_extracted_fields] deterministic mapping found: {mapped}")

        still_missing = [f for f in self._EXTRACTABLE_FIELDS if not mapped.get(f)]
        if still_missing:
            prompt = (
                "Map the following raw document-extraction JSON onto the target schema. "
                f"Focus especially on these fields, which a first pass could not confidently find: {still_missing}. "
                "Only fill a field if you can confidently infer it from the data below. "
                "Leave anything uncertain or absent as an empty string — never guess.\n\n"
                f"RAW EXTRACTION JSON:\n{json.dumps(raw_extraction)}"
            )
            try:
                completion = self.client.beta.chat.completions.parse(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=TravelInsuranceDetails,
                    temperature=0.0,
                )
                parsed = completion.choices[0].message.parsed
                llm_fields = parsed.model_dump() if parsed else {}
                print(f"[parse_extracted_fields] LLM fallback found: {llm_fields}")
                for key in still_missing:
                    val = llm_fields.get(key)
                    if val and str(val).strip():
                        mapped[key] = val
            except Exception as e:
                print(f"[parse_extracted_fields] LLM fallback error (deterministic mapping still applies): {e}")

        print(f"[parse_extracted_fields] final merged fields: {mapped}")
        return mapped

    def extract_document_data(self, file_bytes: bytes, mime_type: str) -> dict:
        """
        Alternative path: extract fields directly from the raw document,
        skipping the Protego extract-pdf call entirely.

        NOT CURRENTLY CALLED by chat.py's upload endpoint (which uses
        QuoteService.extract_pdf_bytes -> parse_extracted_fields instead) —
        this was already true before the OpenAI switch, so it's unused
        either way. Kept for parity.

        IMPORTANT PROVIDER DIFFERENCE: Gemini could accept a raw PDF's bytes
        directly as inline document data. OpenAI's chat completions API does
        NOT accept raw PDF bytes the same way — it expects an image (base64
        data URL). This implementation assumes `file_bytes` is an IMAGE
        (jpg/png), not a PDF. If you need to feed actual multi-page PDFs
        through OpenAI directly (rather than via Protego's own extract-pdf,
        which is what's actually used today), that needs either: converting
        PDF pages to images first, or using OpenAI's Files/Assistants API —
        a separate, larger change from this file. Flagging rather than
        silently guessing, since this is a real capability gap between the
        two providers, not just a syntax difference.
        """
        import base64

        prompt = (
            "You are an AI data extractor for a travel insurance system. "
            "Analyze the provided document image and extract any relevant information matching the "
            "required checklist schema. "
            "If a specific field is not found in the document, you MUST leave it as an empty string ''."
        )

        b64_data = base64.b64encode(file_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{b64_data}"

        try:
            completion = self.client.beta.chat.completions.parse(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": data_url}},
                        ],
                    }
                ],
                response_format=TravelInsuranceDetails,
                temperature=0.0,
            )
            parsed = completion.choices[0].message.parsed
            return parsed.model_dump() if parsed else {}
        except Exception as e:
            print(f"Document Extraction Error: {e}")
            return {}