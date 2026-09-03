"""
Hand-verified, targeted metadata fix for the "types of life insurance"
retrieval bug (2026-09-01). Every correction below was confirmed by reading
the chunk's own full stored text (not just heading/preview) against the
real source PDF pages — no LLM/regex classifier used, per the user's
explicit request to do this manually rather than burn Groq quota on an
automated KB-wide scan.

5 chunks corrected, all from documents whose real body text genuinely
defines a specific life-insurance product but whose metadata excluded it
from retrieval:

1. "Important Types of Life Insurance products" (contains real Term +
   Whole Life definitions) — policy_type was "equity_oriented_insurance_products"
   (wrong; no ULIP/equity content in this chunk at all), corrected to "life".
   section was already correctly "types_of_insurance".
2. "1. Term Insurance plan" — section was "general", corrected to
   "types_of_insurance" (policy_type was already correctly "life").
3. "3) Endowment Assurance" — section was "general", corrected to
   "types_of_insurance" (policy_type already correct).
4. "6) ULIP" — section was "general", corrected to "types_of_insurance"
   (policy_type already correct).
5. The chunk containing "5) Money back assurance product" (heading is a
   garbled bullet-point artifact from chunking, unrelated to this fix) —
   both policy_type ("general") and section ("general") were wrong,
   corrected to "life" / "types_of_insurance".

Same backup-then-overwrite convention as policy_type_retag.py /
section_retag.py.
"""
import json
import time

META_PATH = "/app/app/turbovec_data/documents/insurance_docs_meta.ndjson"

CORRECTIONS = {
    "debba9d1-6083-415c-bdf5-ad16149f1579": {"policy_type": "life"},
    "efdfd244-99b9-4723-bb95-9b666eefd408": {"section": "types_of_insurance"},
    "24012260-05f9-4f9e-a215-162139b3ad0b": {"section": "types_of_insurance"},
    "705eed7e-c57d-450f-9604-02b57a41296a": {"section": "types_of_insurance"},
    "21d0a8b2-a68e-4524-abd3-cc5537d2aca8": {"policy_type": "life", "section": "types_of_insurance"},
}

backup_path = META_PATH + f".bak-{int(time.time())}"
with open(META_PATH, encoding="utf-8") as src, open(backup_path, "w", encoding="utf-8") as dst:
    dst.write(src.read())
print(f"Backed up original to {backup_path}")

lines = []
applied = []
with open(META_PATH, encoding="utf-8") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if not line:
            continue
        d = json.loads(line)
        rid = d.get("id")
        if rid in CORRECTIONS:
            meta = d.setdefault("metadata", {})
            before = {k: meta.get(k) for k in CORRECTIONS[rid]}
            meta.update(CORRECTIONS[rid])
            applied.append((rid, meta.get("section_heading", ""), before, CORRECTIONS[rid]))
            lines.append(json.dumps(d))
        else:
            lines.append(line)

with open(META_PATH, "w", encoding="utf-8") as fh:
    for line in lines:
        fh.write(line + "\n")

print(f"\nApplied {len(applied)} correction(s):")
for rid, heading, before, after in applied:
    print(f"  {rid[:8]}  {heading!r:45}  {before} -> {after}")

missing = set(CORRECTIONS) - {rid for rid, *_ in applied}
if missing:
    print(f"\nWARNING: {len(missing)} id(s) not found in the file: {missing}")
