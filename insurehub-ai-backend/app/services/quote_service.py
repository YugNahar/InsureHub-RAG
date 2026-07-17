# app/services/quote_service.py
import requests
import json
from app.schemas.travel import get_coverage_type_code, MAX_TRAVELLERS_TOTAL


def build_travellers_list(extracted_data: dict) -> list:
    """
    Builds the "travellers" array Protego expects: the primary traveller
    (from top-level first_name/last_name/date_of_birth) plus any companions
    collected for Group/Family policies (extracted_data["additional_travellers"],
    a list of {first_name, last_name, date_of_birth} dicts). Individual
    policies naturally end up with exactly 1 entry since there are no
    companions to append. Hard-capped at MAX_TRAVELLERS_TOTAL as a safety net.

    Matches the get-quotes schema exactly (first_name/last_name/date_of_birth
    only — no extra "name" field, in case the API validates strictly against
    unknown properties).
    """
    first_name = extracted_data.get("first_name", "").strip()
    last_name = extracted_data.get("last_name", "").strip()

    travellers = [{
        "first_name": first_name,
        "last_name": last_name,
        "date_of_birth": extracted_data.get("date_of_birth", ""),
    }]

    for companion in extracted_data.get("additional_travellers", []):
        travellers.append({
            "first_name": (companion.get("first_name") or "").strip(),
            "last_name": (companion.get("last_name") or "").strip(),
            "date_of_birth": companion.get("date_of_birth", ""),
        })

    return travellers[:MAX_TRAVELLERS_TOTAL]


class QuoteService:
    # Pointing to the travel router prefix based on the API architecture
    # Updated 2026-07 to the new UAT environment (was insurance-backend-fastapi-dev-...).
    BASE_URL = "https://uat-insure-hub-service-cwemhvcfd3habvf5.centralindia-01.azurewebsites.net/api/v1/travel"

    # Separate prefix — create-session lives under /users/, not /travel/.
    USERS_BASE_URL = "https://uat-insure-hub-service-cwemhvcfd3habvf5.centralindia-01.azurewebsites.net/api/v1/users"

    @staticmethod
    def _split_mobile_number(mobile_number: str, default_country_code: str = "971") -> tuple:
        """
        Our session state stores mobile_number as one string, sometimes with a
        '+971' prefix already in it (e.g. "+971 500024681"), but create-session
        wants country_code and mobile_number as two SEPARATE fields, with no
        '+' (the real InsureHub frontend sends country_code: "971", not "+971").
        Strips a leading '+' and/or the country code if present; otherwise
        assumes the whole string is the local number and defaults the code to
        UAE, matching the '+971' default already assumed elsewhere in this file.
        """
        raw = (mobile_number or "").strip().replace(" ", "").replace("-", "")
        if raw.startswith("+"):
            raw = raw[1:]
        if raw.startswith(default_country_code):
            return default_country_code, raw[len(default_country_code):]
        return default_country_code, raw

    @staticmethod
    def create_session(extracted_data: dict) -> dict:
        """
        Calls Protego's create-session endpoint. Returns a dict:
            {"session_id": "3169", "client_id": "1592"}
        or {} if either couldn't be obtained.

        Confirmed real response shape (2026-07-15 test): both IDs are nested,
        not top-level:
            data["session_data"]["id"]        -> session_id
            data["client"]["id"]              -> client_id
            (data["session_data"]["client_id"] is the same value, as a fallback)

        IMPORTANT: client_id is NOT a fixed platform constant. Protego creates
        a fresh client record per create-session call and returns its own id.
        The "1618" hardcoded everywhere else in this file before this was a
        guess that happened not to error — it should be replaced with this
        real per-session value, same as session_id.
        """
        url = f"{QuoteService.USERS_BASE_URL}/create-session"
        headers = {"Content-Type": "application/json"}

        country_code, local_number = QuoteService._split_mobile_number(
            extracted_data.get("mobile_number", "")
        )

        payload = {
            "step": "step_1_pre_request",
            "assigned_user_id": None,
            "payload": {
                "first_name": extracted_data.get("first_name", "").strip(),
                "last_name": extracted_data.get("last_name", "").strip(),
                "country_code": country_code,
                "mobile_number": local_number,
                "email": extracted_data.get("email", "").strip(),
                "partner_code": "",
                "friends_and_family_contact": "",
                "marketing_consent": extracted_data.get("marketing_consent", "no"),
            },
            "lob_id": 1,
        }

        print(f"[create_session] outgoing payload: {json.dumps(payload)}")

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            print(f"[create_session] status: {response.status_code}")
            print(f"[create_session] raw response (first 2000 chars): {response.text[:2000]}")
            response.raise_for_status()

            data = response.json()
            session_data = data.get("session_data") or {}
            client = data.get("client") or {}

            session_id = session_data.get("id")
            client_id = client.get("id") or session_data.get("client_id")

            if session_id is None or client_id is None:
                print(f"[create_session] WARNING: couldn't find session_id/client_id in response shape: {data}")
                return {}

            print(f"[create_session] obtained session_id={session_id}, client_id={client_id}")
            return {"session_id": str(session_id), "client_id": str(client_id)}
        except requests.exceptions.RequestException as e:
            print(f"[create_session] Protego Create Session API error: {e}")
            return {}

    @staticmethod
    def fetch_live_quotes(extracted_data: dict) -> list:
        """
        Maps conversational session state into the deeply nested JSON payload
        required by Protego's get-quotes endpoint.

        NOTE: keys here MUST match the canonical field names used in
        app/schemas/travel.py (TravelInsuranceDetails) and the checklist in
        chat.py, since that's what actually gets written into session state.
        """
        url = f"{QuoteService.BASE_URL}/get-quotes"

        first_name = extracted_data.get("first_name", "").strip()
        last_name = extracted_data.get("last_name", "").strip()

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream"
        }

        payload = {
            "lob_id": 1,
            "personal_details": {
                "first_name": first_name,
                "last_name": last_name,
                "email": extracted_data.get("email", ""),
                "friends_and_family_contact": "",
                "marketing_consent": "no",
                "mobile_country_code": "+971",
                "mobile_number": extracted_data.get("mobile_number", ""),
                "partner_code": ""
            },
            "travel_details": {
                "cover_type": extracted_data.get("cover_type", "Individual").capitalize(),
                "coverage_type": get_coverage_type_code(extracted_data.get("coverage_type", "")),
                "departure": extracted_data.get("departure") or "United Arab Emirates",
                "destination": extracted_data.get("destination", ""),
                "plan_type": extracted_data.get("plan_type", "single").lower(),
                "travel_dates": {
                    "start_date": extracted_data.get("start_date", ""),
                    "end_date": extracted_data.get("end_date", "")
                },
                "travellers": build_travellers_list(extracted_data)
            }
        }

        print(f"[fetch_live_quotes] outgoing payload: {json.dumps(payload)}")

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            print(f"[fetch_live_quotes] status: {response.status_code}, content-type: {response.headers.get('Content-Type')}")
            print(f"[fetch_live_quotes] raw response (first 3000 chars): {response.text[:3000]}")
            response.raise_for_status()

            # Handling Server-Sent Events (SSE) format vs clean JSON list parsing
            raw_json_data = []
            for line in response.text.splitlines():
                if line.startswith("data: "):
                    if "completed" in line:
                        continue
                    clean_json_str = line.replace("data: ", "").strip()
                    if clean_json_str:
                        raw_json_data.append(json.loads(clean_json_str))

            if not raw_json_data and response.headers.get("Content-Type") == "application/json":
                return response.json()

            if not raw_json_data:
                print("[fetch_live_quotes] parsed zero quote events from response — see raw response above")

            return raw_json_data
        except requests.exceptions.RequestException as e:
            print(f"Protego Get Quotes API error: {e}")
            return []

    @staticmethod
    def bind_quote(extracted_data: dict, quote_index: int) -> dict:
        """
        Calls Protego's bind-quotes endpoint to lock in a specific policy selection.

        NOTE: same field-name caveat as fetch_live_quotes applies here (full_name /
        gender / nationality / city_of_residence are not currently collected anywhere
        in the chat flow, so they still fall back to placeholders). Flagging this as
        a follow-up item — out of scope for the upload/missing-fields/get-quotes fix.
        """
        url = f"{QuoteService.BASE_URL}/bind-quotes"
        headers = {"Content-Type": "application/json"}

        available_quotes = extracted_data.get("available_quotes", [])
        selected_quote = available_quotes[quote_index] if len(available_quotes) > quote_index else {}

        first_name = extracted_data.get("first_name", "").strip()
        last_name = extracted_data.get("last_name", "").strip()
        full_name = f"{first_name} {last_name}".strip() or "Valued Guest"

        # Quotes come back nested as {"insurer": {"id","name"}, "plan": {"id","name","price"}, ...}
        # per how chat.py already displays them. Support both nested and flat shapes defensively.
        insurer_id = selected_quote.get("insurer", {}).get("id") or selected_quote.get("insurer_id", 0)
        plan = selected_quote.get("plan", {})
        plan_id = plan.get("id") or selected_quote.get("plan_id", "")
        plan_name = plan.get("name") or selected_quote.get("plan_name", "Standard")
        quote_id = selected_quote.get("quote_id") or selected_quote.get("id", "")

        payload = {
            "insurerId": insurer_id,
            "plan_name": plan_name,
            "action": "bind",
            "session_id": extracted_data.get("protego_session_id", ""),
            "client_id": 1618,
            "quoteId": quote_id,
            "lob_id": 1,
            "insurerPlan": {
                "id": plan_id,
                "planName": plan_name
            },
            "travellers_more_info": [
                {
                    "first_name": first_name,
                    "last_name": last_name,
                    "date_of_birth": extracted_data.get("date_of_birth", "2001-09-21"),
                    "gender": extracted_data.get("gender", "Male"),
                    "nationality": extracted_data.get("nationality", "United Arab Emirates"),
                    "cityOfResidence": extracted_data.get("city_of_residence", "Dubai"),
                    "sourceOfFunds": "Salary",
                    "beneficiaryName": "",
                    "beneficiaryRelationship": "",
                    "address": "",
                    "passportNo": extracted_data.get("passport_number", "P1234567"),
                    "passportIssueDate": "",
                    "passportExpiryDate": "",
                    "emiratesId": extracted_data.get("emirates_id", "784-1995-1234567-1"),
                    "emiratesIdIssueDate": "",
                    "emiratesIdExpiryDate": ""
                }
            ],
            "contact": {
                "mobileCountryCode": "+971",
                "mobileNumber": extracted_data.get("mobile_number", ""),
                "email": extracted_data.get("email", "user@example.com")
            },
            "identity": {
                "emiratesId": extracted_data.get("emirates_id", "784-1995-1234567-1"),
                "passportNo": extracted_data.get("passport_number", "P1234567"),
                "emiratesIdExpiryDate": "",
                "emiratesIdIssueDate": "",
                "passportExpiryDate": "",
                "passportIssueDate": ""
            },
            "declarations": {
                "isUaeResidentConfirmed": True
            },
            "attachments": []
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Protego Bind Quote API error: {e}")
            return {}

    @staticmethod
    def issue_policy(booking_data: dict) -> dict:
        """
        Calls Protego's issue-policy endpoint to finalize coverage
        and render legal policy documentation.
        """
        url = f"{QuoteService.BASE_URL}/issue-policy"
        headers = {"Content-Type": "application/json"}

        payload = {
            "insurerId": booking_data.get("insurer_id", 0),
            "action": "issue",
            "session_id": booking_data.get("session_id") or booking_data.get("protego_session_id", ""),
            "client_id": 1618,
            "user_id": booking_data.get("user_id", 0),
            "plan_name": booking_data.get("plan_name", "Standard"),
            "provider_transaction_id": booking_data.get("provider_transaction_id", ""),
            "lob_id": 1,
            "insurer_policy_number": booking_data.get("insurer_policy_number", ""),
            "start_date": booking_data.get("start_date", "2026-07-09T11:22:14.890Z"),
            "end_date": booking_data.get("end_date", "2026-07-09T11:22:14.890Z"),
            "policy_issued_by": booking_data.get("policy_issued_by", "system"),
            "policy_status": booking_data.get("policy_status", "Issued")
        }

        try:
            response = requests.post(url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Protego Issue Policy API error: {e}")
            return {}

    @staticmethod
    def extract_pdf(file_path: str) -> dict:
        """
        Calls the extract-pdf endpoint to parse travel documents from a file on disk.
        """
        url = f"{QuoteService.BASE_URL}/extract-pdf"

        form_data = {
            "ai_engine": "openai",
            "ocr_engine": "tesseract",
            "insurance_type": "Travel"
        }

        try:
            with open(file_path, 'rb') as pdf_file:
                upload_files = {
                    'files': (file_path.split('/')[-1], pdf_file, 'application/pdf')
                }
                response = requests.post(url, data=form_data, files=upload_files, timeout=30)
                response.raise_for_status()
                return response.json()
        except FileNotFoundError:
            print(f"Error: Could not find PDF file at {file_path}")
            return {}
        except requests.exceptions.RequestException as e:
            print(f"Protego Extract PDF API error: {e}")
            return {}

    @staticmethod
    def extract_pdf_bytes(file_bytes: bytes, filename: str, content_type: str = None) -> dict:
        """
        Same as extract_pdf, but takes raw bytes directly (e.g. from a FastAPI
        UploadFile) so the backend doesn't need to write to disk first.
        """
        url = f"{QuoteService.BASE_URL}/extract-pdf"

        form_data = {
            "ai_engine": "openai",
            "ocr_engine": "tesseract",
            "insurance_type": "Travel"
        }

        try:
            upload_files = {
                'files': (filename, file_bytes, content_type or 'application/octet-stream')
            }
            response = requests.post(url, data=form_data, files=upload_files, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Protego Extract PDF API error: {e}")
            return {}