import json
import re
import collections

path = "/app/app/turbovec_data/documents/insurance_docs_meta.ndjson"

EMAIL_RE = re.compile(r"[^\s,;@]+@[^\s,;@]+\.[^\s,;@]+")

sources = collections.defaultdict(list)
for line in open(path):
    d = json.loads(line)
    md = d.get("metadata", {})
    src = md.get("source", "?")
    sources[src].append(d.get("text", ""))

found_any = False
for src in sorted(sources):
    texts = sources[src]
    full_text = " ".join(texts)
    emails = sorted(set(e.lower() for e in EMAIL_RE.findall(full_text)))
    is_csv_name = src.lower().endswith(".csv")
    if is_csv_name or len(emails) >= 3:
        found_any = True
        readable = re.sub(r"^[0-9a-f]{8,}_", "", src)
        print(f"{readable} | source={src} | chunks={len(texts)} | distinct_emails={len(emails)} | csv_extension={is_csv_name}")
        if emails:
            print("   emails:", emails[:10])

if not found_any:
    print("No CSV-named documents or email-list-shaped content found in the KB.")
