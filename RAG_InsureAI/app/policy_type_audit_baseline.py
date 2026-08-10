"""
Standing audit for chunk `policy_type` metadata drift.

Deterministic, code-only checks (no LLM calls) that flag chunks worth a
human/gold-set look. Companion to contamination_corpus_runner.py — same
purpose (a standing regression gate), different layer (KB tagging vs.
generated-answer contamination).

Two signals, both explained in plan_policy_type_tagging.md:

1. Minority-class contamination inside single-topic documents — a chunk
   tagged something other than its document's own dominant type (or
   "general") is suspicious. Scoped to documents confirmed single-topic
   by source filename; the two large multi-topic references (the law
   textbook and the handbook) are excluded from this specific check —
   see SINGLE_TOPIC_DOCS below.

2. `policy_type not in all_policy_types` — a chunk's own tag disagrees
   with the document-level keyword-hit list computed at whole-document
   tag time (tag_document(), metadata_tagger.py). This is NOT proof of
   a mistag on its own (a genuinely single-topic chunk inside a
   multi-topic document can legitimately disagree with the document's
   aggregate stats) — treat it as a review flag, never as grounds to
   auto-correct.

Do NOT gate on policy_type_confidence: confirmed live (this investigation)
that field is a DOCUMENT-level value leaked onto every chunk from that
document (uniform per source file), not a per-chunk confidence — see
RC-5 in plan_policy_type_tagging.md. A confidence floor would do nothing.

Usage:
    python3 policy_type_audit.py                      # human-readable report
    python3 policy_type_audit.py --json                # machine-readable
    python3 policy_type_audit.py --max-minority N       # exit non-zero above N (default 5)
    python3 policy_type_audit.py --max-disagree N        # exit non-zero above N (default 45)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_META_PATH = os.path.join(
    os.path.dirname(__file__), "app", "turbovec_data", "documents", "insurance_docs_meta.ndjson"
)

# Source filename -> the ONE policy_type this document is genuinely about.
# A chunk from one of these documents tagged anything else (and not
# "general") is minority-class contamination — see plan_policy_type_tagging.md
# RC-2/RC-3 for the confirmed failure shape this catches.
#
# Deliberately does NOT include the two large multi-topic references
# ("1f2e36d3fb97_9.3 INSURANCE LAW AND PRACTICE.pdf" and
# "insurance hb 1101.pdf") — those genuinely span many types and a
# document-topic check would itself be wrong for them.
SINGLE_TOPIC_DOCS: dict[str, str] = {
    "5e9acf857576_travelinsuranceguide.pdf": "travel",
    "ea22bdd3a9bf_m4-5f.pdf": "health",
    "df12091601c0_m3-f2.pdf": "life",
    "5a1c9b6aaef2_m4-7f.pdf": "liability",
    "b9cc853485a4_m4-3f.pdf": "motor",
    "015ed1fc1d47_Pet_Insurance_Guide.pdf": "health",
}


def load_chunks(meta_path: str) -> list[dict]:
    chunks = []
    with open(meta_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return chunks


def audit(chunks: list[dict]) -> dict:
    minority: list[dict] = []
    disagree: list[dict] = []
    per_source_dist: dict[str, dict[str, int]] = {}

    for d in chunks:
        m = d.get("metadata", {}) or {}
        src = m.get("source", "?")
        pt = m.get("policy_type", "?")
        per_source_dist.setdefault(src, {})
        per_source_dist[src][pt] = per_source_dist[src].get(pt, 0) + 1

        expected = SINGLE_TOPIC_DOCS.get(src)
        if expected and pt not in (expected, "general"):
            minority.append({
                "id": d.get("id"), "source": src, "tagged": pt, "expected_doc_topic": expected,
                "text_preview": (d.get("text", "") or "")[:200],
            })

        all_types_raw = m.get("all_policy_types")
        if all_types_raw and pt != "general":
            all_types = [x.strip() for x in str(all_types_raw).split(",") if x.strip()]
            if all_types and pt not in all_types:
                disagree.append({
                    "id": d.get("id"), "source": src, "tagged": pt,
                    "all_policy_types": all_types,
                })

    return {
        "total_chunks": len(chunks),
        "per_source_distribution": per_source_dist,
        "minority_class_contamination": minority,
        "policy_type_vs_all_policy_types_disagreement": disagree,
    }


def print_report(result: dict) -> None:
    print("=" * 70)
    print("POLICY_TYPE AUDIT")
    print("=" * 70)
    print(f"Total chunks: {result['total_chunks']}")
    print()
    print("-- Per-source policy_type distribution --")
    for src, dist in sorted(result["per_source_distribution"].items()):
        total = sum(dist.values())
        print(f"{total:>4}  {src}")
        for pt, c in sorted(dist.items(), key=lambda x: -x[1]):
            marker = " <-- expected" if SINGLE_TOPIC_DOCS.get(src) == pt else ""
            print(f"         {c:>4}  {pt}{marker}")
    print()

    minority = result["minority_class_contamination"]
    print(f"-- Minority-class contamination in single-topic docs: {len(minority)} --")
    for m in minority[:20]:
        print(f"  {m['id'][:8]}  {m['source'][:35]:<35}  tagged={m['tagged']:<12} expected={m['expected_doc_topic']}")
    if len(minority) > 20:
        print(f"  ... and {len(minority) - 20} more")
    print()

    disagree = result["policy_type_vs_all_policy_types_disagreement"]
    print(f"-- policy_type not in all_policy_types (review flag, not proof of error): {len(disagree)} --")
    for d in disagree[:20]:
        print(f"  {d['id'][:8]}  {d['source'][:35]:<35}  tagged={d['tagged']:<18} all=[{', '.join(d['all_policy_types'])}]")
    if len(disagree) > 20:
        print(f"  ... and {len(disagree) - 20} more")
    print("=" * 70)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--meta-path", default=DEFAULT_META_PATH)
    p.add_argument("--json", action="store_true")
    p.add_argument("--max-minority", type=int, default=5,
                    help="exit non-zero if minority-class contamination exceeds this")
    p.add_argument("--max-disagree", type=int, default=45,
                    help="exit non-zero if all_policy_types disagreement exceeds this")
    args = p.parse_args()

    chunks = load_chunks(args.meta_path)
    result = audit(chunks)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)

    n_minority = len(result["minority_class_contamination"])
    n_disagree = len(result["policy_type_vs_all_policy_types_disagreement"])
    ok = n_minority <= args.max_minority and n_disagree <= args.max_disagree
    if not args.json:
        verdict = "PASS" if ok else "FAIL"
        print(f"\n{verdict} — minority={n_minority} (max {args.max_minority}), "
              f"disagree={n_disagree} (max {args.max_disagree})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
