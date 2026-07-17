"""
Standalone diagnostic for the get-quotes extraction failure.

Run this directly in your backend environment (same venv/container as the
FastAPI app):

    python debug_extraction.py

It isolates the exact Gemini call used by extract_fields_from_conversation —
no FastAPI, no database, no frontend involved — so whatever error shows up
here IS the error breaking your chat flow. Paste the full output back and
that tells us definitively what's wrong, instead of guessing further.
"""

import sys
import traceback

print(f"Python: {sys.version}")

try:
    import google.generativeai as genai
    print(f"google-generativeai version: {getattr(genai, '__version__', 'UNKNOWN')}")
except Exception:
    print("FAILED to import google.generativeai — is it installed in this environment?")
    traceback.print_exc()
    sys.exit(1)

# --- Load your API key the same way the app does -----------------------
# Adjust this import if your settings module lives elsewhere.
try:
    from app.core.config import settings
    api_key = settings.GEMINI_API_KEY
    print(f"Loaded GEMINI_API_KEY from app.core.config.settings (length {len(api_key or '')})")
except Exception:
    print("Could not import app.core.config.settings — falling back to env var GEMINI_API_KEY")
    import os
    api_key = os.environ.get("GEMINI_API_KEY")
    print(f"GEMINI_API_KEY from env (length {len(api_key or '')})")

if not api_key:
    print("\nNO API KEY FOUND. This alone would explain every call failing.")
    sys.exit(1)

genai.configure(api_key=api_key)

# --- Minimal reproduction of TravelInsuranceDetails ----------------------
from pydantic import BaseModel, Field

class TravelInsuranceDetailsProbe(BaseModel):
    first_name: str = Field(default="")
    last_name: str = Field(default="")
    email: str = Field(default="")
    mobile_number: str = Field(default="")
    coverage_type: str = Field(default="")
    destination: str = Field(default="")
    departure: str = Field(default="")
    start_date: str = Field(default="")
    end_date: str = Field(default="")
    plan_type: str = Field(default="")
    cover_type: str = Field(default="")
    date_of_birth: str = Field(default="")

print("\n--- TEST 1: minimal schema, minimal prompt ---")
try:
    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    response = model.generate_content(
        contents=[{"role": "user", "parts": ["My name is Sarah Jenkins, dob 1990-09-09"]}],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=TravelInsuranceDetailsProbe,
            temperature=0.0,
        ),
    )
    print("SUCCESS. Raw response.text:")
    print(response.text)
except Exception:
    print("FAILED. Full traceback:")
    traceback.print_exc()

print("\n--- TEST 2: exact prompt from a real 'still missing' turn ---")
try:
    from app.schemas.travel import TravelInsuranceDetails

    prompt = (
        "Extract travel-insurance details from the user's message below. "
        "Focus especially on these fields, which are still missing: "
        "['plan_type', 'cover_type', 'date_of_birth']. "
        "Only fill a field if you can confidently determine it from this message. "
        "Leave anything uncertain or not mentioned as an empty string — never guess.\n\n"
        "USER MESSAGE:\nPlan type is single, cover type is Individual and dob is 21/09/2001"
    )

    model = genai.GenerativeModel(model_name='gemini-1.5-flash')
    response = model.generate_content(
        contents=[{"role": "user", "parts": [prompt]}],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=TravelInsuranceDetails,
            temperature=0.0,
        ),
    )
    print("SUCCESS. Raw response.text:")
    print(response.text)
except Exception:
    print("FAILED. Full traceback:")
    traceback.print_exc()

print("\n--- DONE ---")
print("If TEST 1 failed: the problem is your google-generativeai version/setup,")
print("  not anything specific to this app's schema. Try: pip install -U google-generativeai")
print("If TEST 1 succeeded but TEST 2 failed: something about the real")
print("  TravelInsuranceDetails schema itself is the problem — paste both outputs.")
print("If both succeeded: the bug is downstream (state merge, checklist, session")
print("  handling) rather than the Gemini call — paste the printed JSON from TEST 2")
print("  and I'll look at what happens to it after extraction.")