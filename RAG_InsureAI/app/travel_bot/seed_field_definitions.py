# Run this ONCE from your project root (same folder as your app/ directory):
#     python seed_field_definitions.py
#
# Populates field_definitions for request_type="Travel Insurance" per the list
# your senior sent. Idempotent — safe to re-run (skips fields that already
# exist for this request_type).
#
# IMPORTANT — read before running:
# The "COLLECTED_TODAY" column below is honest about what actually happens
# right now: fields marked False are NOT asked anywhere in the current chat
# flow or extracted by Gemini (see app/schemas/travel.py's
# TravelInsuranceDetails and app/routers/chat.py's REQUIRED_FIELDS/
# FIELD_QUESTIONS). Seeding the definition does not make the bot start
# collecting it — that needs the extraction schema + conversation flow
# extended separately. This script only builds the CHECKLIST TEMPLATE;
# it's a deliberate, visible gap until that follow-up work happens.

from sqlalchemy.orm import Session
from travel_bot.core.database import SessionLocal, Base, engine
from travel_bot.models.field_definition import FieldDefinition

REQUEST_TYPE = "Travel Insurance"

# (field_key, display_name, data_type, is_required, validation_rules_json, display_order, COLLECTED_TODAY)
FIELD_DEFINITIONS = [
    ("full_name",                    "Full Name",                    "string", True,  None,                                 1,  True),
    ("date_of_birth",                "Date of Birth",                "date",   True,  '{"format": "YYYY-MM-DD"}',           2,  True),
    ("gender",                       "Gender",                       "enum",   True,  '{"options": ["Male", "Female", "Other"]}', 3, False),
    ("nationality",                  "Nationality",                  "string", True,  None,                                 4,  False),
    ("passport_number",              "Passport Number",              "string", True,  None,                                 5,  False),
    ("passport_expiry_date",         "Passport Expiry Date",         "date",   True,  '{"format": "YYYY-MM-DD"}',           6,  False),
    ("mobile_number",                "Mobile Number",                "string", True,  None,                                 7,  True),
    ("email_address",                "Email Address",                "string", True,  '{"format": "email"}',                8,  True),
    ("country_of_residence",         "Country of Residence",         "string", True,  None,                                 9,  True),
    ("destination_country",          "Destination Country/Countries","string", True,  None,                                 10, True),
    ("trip_type",                    "Trip Type",                    "enum",   True,  '{"options": ["Single Trip", "Annual Multi-Trip"]}', 11, True),
    ("departure_date",               "Departure Date",               "date",   True,  '{"format": "YYYY-MM-DD"}',           12, True),
    ("return_date",                  "Return Date",                  "date",   True,  '{"format": "YYYY-MM-DD"}',           13, True),
    ("trip_duration",                "Trip Duration",                "number", False, '{"note": "derivable from departure/return dates"}', 14, False),
    ("number_of_travelers",          "Number of Travelers",          "number", True,  None,                                 15, True),
    ("traveler_details",             "Traveler Details",             "list",   False, '{"required_if": "cover_type in [Group, Family]"}', 16, True),
    ("purpose_of_travel",            "Purpose of Travel",            "enum",   False, '{"options": ["Business", "Leisure", "Study"]}', 17, False),
    ("sum_insured",                  "Sum Insured",                  "number", False, None,                                 18, False),
    ("pre_existing_medical_conditions", "Pre-existing Medical Conditions", "string", False, None,                           19, False),
    ("visa_type",                    "Visa Type",                    "string", False, None,                                 20, False),
    ("occupation",                   "Occupation",                   "string", False, None,                                 21, False),
    ("nominee_name",                 "Nominee Name",                 "string", False, None,                                 22, False),
    ("nominee_relationship",         "Relationship with Nominee",    "string", False, None,                                 23, False),
    ("address",                      "Address",                     "string", False, None,                                 24, False),
    ("postal_code",                  "Postal Code",                  "string", False, None,                                 25, False),
    ("emergency_contact_name",       "Emergency Contact Name",       "string", False, None,                                 26, False),
    ("emergency_contact_number",     "Emergency Contact Number",     "string", False, None,                                 27, False),
    # Insurer/product-specific fields this bot already needs that weren't in
    # the generic list — covered by "Any other insurer-specific mandatory
    # fields" in the brief.
    ("coverage_type",                "Coverage Type (Insurance Product)", "enum", True, '{"options": ["Hajj and Umrah", "UAE Inbound", "Worldwide", "Schengen", "GCC Countries"]}', 28, True),
    ("cover_type",                   "Cover Type (Who's Insured)",   "enum",   True,  '{"options": ["Individual", "Group", "Family"]}', 29, True),
]


def seed():
    Base.metadata.create_all(bind=engine)  # creates field_definitions table if missing
    db: Session = SessionLocal()
    try:
        existing_keys = {
            row.field_key
            for row in db.query(FieldDefinition).filter(FieldDefinition.request_type == REQUEST_TYPE).all()
        }
        added, skipped = 0, 0
        for field_key, display_name, data_type, is_required, validation_rules, display_order, collected_today in FIELD_DEFINITIONS:
            if field_key in existing_keys:
                skipped += 1
                continue
            db.add(FieldDefinition(
                request_type=REQUEST_TYPE,
                field_key=field_key,
                display_name=display_name,
                data_type=data_type,
                is_required=is_required,
                validation_rules=validation_rules,
                display_order=display_order,
            ))
            added += 1
            if not collected_today:
                print(f"  NOTE: '{field_key}' seeded as a definition, but the bot does NOT collect it yet.")
        db.commit()
        print(f"\nDone — added {added}, skipped {skipped} already-existing definitions for '{REQUEST_TYPE}'.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()