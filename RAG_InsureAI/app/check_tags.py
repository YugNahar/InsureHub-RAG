import json
import re
import collections

path = "/app/app/turbovec_data/documents/insurance_docs_meta.ndjson"

sources = collections.defaultdict(list)
for line in open(path):
    d = json.loads(line)
    md = d.get("metadata", {})
    src = md.get("source", "?")
    tag = md.get("policy_type", "?")
    sources[src].append(tag)

rx = r"^[0-9a-f]{8,}_(.+?)_insurance_guide(?:_v\d+)?\.pdf$"
pattern = re.compile(rx, re.IGNORECASE)

for src in sorted(sources):
    types = sources[src]
    counts = collections.Counter(types)
    readable = re.sub(r"^[0-9a-f]{8,}_", "", src)
    m = pattern.match(src)
    flag = ""
    if m:
        expected = m.group(1).lower()
        off_count = 0
        for k, v in counts.items():
            kl = k.lower()
            if expected not in kl and kl not in expected:
                off_count += v
        if off_count > len(types) / 2:
            flag = "  <-- {}/{} chunks off, expected: {}".format(
                off_count, len(types), expected
            )
    print(readable, "| n=", len(types), "|", dict(counts), flag)
