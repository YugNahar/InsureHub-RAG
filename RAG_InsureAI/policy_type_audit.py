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
import re
import sys

DEFAULT_META_PATH = os.path.join(
    os.path.dirname(__file__), "app", "turbovec_data", "documents", "insurance_docs_meta.ndjson"
)

# Phase 4 of plan_claim_answer_correctness.md: the chunker's own `section`
# field has no "claims" bucket at all (confirmed live — the law/practice
# handbook's chunks only ever get legislation/general/types_of_insurance/
# principles/definitions/history/case_law/chapter). Claims-relevant chunks
# have to be found by vocabulary instead. Deliberately generic across
# insurance types (claim form, survey report, settlement, assessment,
# third party, no claim bonus, discharge...) rather than motor/life-
# specific, since the whole point of this audit is to see how ALL types'
# claims content is distributed/tagged in the shared multi-topic chapters.
_CLAIMS_VOCAB_RE = re.compile(
    r"\bclaim\s+form\b|\bsurvey\s+report\b|\bclaim\s+settlement\b|\bsettl\w*\s+the\s+claim\b|"
    r"\bthird\s+party\s+claims?\b|\bno\s+claim\s+bonus\b|\bloss\s+assessor\b|\bsurveyor\b|"
    r"\bdischarge\s+(?:voucher|form)\b|\bclaims?\s+procedure\b|\bclaims?\s+intimation\b|"
    r"\btotal\s+loss\b|\bsalvage\b|\bclaim\s+documents?\b|\bassessment\s+of\s+(?:the\s+)?claim\b|"
    r"\brepudiat\w*\s+the\s+claim\b|\bcondition\s+of\s+average\b",
    re.IGNORECASE,
)

# Documents known (from this session's live investigation) to mix claims
# content for several policy types in one continuous run of pages — the
# two large multi-topic references, deliberately excluded from
# SINGLE_TOPIC_DOCS above for the same reason.
CLAIMS_SCOPE_DOCS = (
    "1f2e36d3fb97_9.3 INSURANCE LAW AND PRACTICE.pdf",
    "insurance hb 1101.pdf",
)


# A table-of-contents/index chunk (chapter title + "..." + page number,
# repeated) can trip _CLAIMS_VOCAB_RE on a heading's own words ("Claims
# Procedure in Respect of a Life Insurance Policy … 232") without containing
# any actual claims-body content. Confirmed live: an unfiltered first pass
# of this audit showed 31/53 matches tagged "general" with all_policy_types
# always exactly [life, motor] — every one of those turned out to be a ToC
# page, not a body chunk, which would have produced a false "claims content
# is mostly general-tagged" conclusion. 3+ ellipsis-leader-to-number
# sequences is a reliable signal for this document's specific ToC layout.
_TOC_LEADER_RE = re.compile(r"[.…]{2,}\s*\d{1,4}\b")


def _looks_like_toc(text: str) -> bool:
    return len(_TOC_LEADER_RE.findall(text)) >= 3


def audit_claims_scope(chunks: list[dict]) -> dict:
    """Phase 4: for chunks whose text matches claims vocabulary AND whose
    source is one of the known multi-topic claims-chapter documents,
    report the policy_type / all_policy_types distribution. This is NOT
    the same check as `audit()` above (that one flags a chunk's tag
    DISAGREEING with something) — this one just describes what's there,
    so a human (or Phase 1) can decide whether a chunk that legitimately
    covers many types should stay "general" with all_policy_types
    populated, vs. a chunk that's really single-type but mistagged
    "general" and should be retagged.
    """
    matched: list[dict] = []
    toc_skipped = 0
    type_dist: dict[str, int] = {}
    all_types_seen: dict[str, int] = {}
    for d in chunks:
        m = d.get("metadata", {}) or {}
        src = m.get("source", "?")
        if src not in CLAIMS_SCOPE_DOCS:
            continue
        text = d.get("text", "") or ""
        if not _CLAIMS_VOCAB_RE.search(text):
            continue
        if _looks_like_toc(text):
            toc_skipped += 1
            continue
        pt = m.get("policy_type", "?")
        all_types_raw = m.get("all_policy_types") or ""
        all_types = [x.strip() for x in str(all_types_raw).split(",") if x.strip()]
        type_dist[pt] = type_dist.get(pt, 0) + 1
        for t in all_types:
            all_types_seen[t] = all_types_seen.get(t, 0) + 1
        matched.append({
            "id": d.get("id"), "source": src, "page": m.get("page"),
            "tagged": pt, "all_policy_types": all_types,
            "text_preview": text[:180].replace("\n", " "),
        })
    return {
        "claims_scope_chunk_count": len(matched),
        "toc_chunks_skipped": toc_skipped,
        "policy_type_distribution": dict(sorted(type_dist.items(), key=lambda x: -x[1])),
        "all_policy_types_mentions": dict(sorted(all_types_seen.items(), key=lambda x: -x[1])),
        "chunks": matched,
    }


def print_claims_scope_report(result: dict) -> None:
    print("=" * 70)
    print("CLAIMS-SCOPE AUDIT (Phase 4, plan_claim_answer_correctness.md)")
    print("=" * 70)
    print(f"Claims-vocabulary chunks found in {CLAIMS_SCOPE_DOCS}: {result['claims_scope_chunk_count']}  "
          f"(+ {result['toc_chunks_skipped']} table-of-contents chunks excluded)")
    print()
    print("-- policy_type distribution among claims-vocabulary chunks --")
    for pt, c in result["policy_type_distribution"].items():
        print(f"  {c:>4}  {pt}")
    print()
    print("-- all_policy_types mentions (a chunk can appear in several) --")
    for t, c in result["all_policy_types_mentions"].items():
        print(f"  {c:>4}  {t}")
    print()
    print("-- Sample chunks (first 15) --")
    for c in result["chunks"][:15]:
        print(f"  {c['id'][:8]}  p{c['page']:<4}  tagged={c['tagged']:<10}  "
              f"all=[{', '.join(c['all_policy_types'])}]")
        print(f"      {c['text_preview']}")
    if len(result["chunks"]) > 15:
        print(f"  ... and {len(result['chunks']) - 15} more")
    print("=" * 70)

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
    "801b5f761448_travel_insurance_guide.pdf": "travel",
    "ea22bdd3a9bf_m4-5f.pdf": "health",
    "df12091601c0_m3-f2.pdf": "life",
    "5a1c9b6aaef2_m4-7f.pdf": "liability",
    "b9cc853485a4_m4-3f.pdf": "motor",
    # Was "health" -- stale from before pet_insurance existed as its own
    # open-vocab type (project_open_vocab_promotion_wired). This document
    # IS about pet insurance; "health" was never the right expectation
    # for it, just the closest type available at the time this table was
    # written. Confirmed live 2026-08-10: 12/12 chunks consistently tagged
    # pet_insurance, which is correct, not contamination.
    "015ed1fc1d47_Pet_Insurance_Guide.pdf": "pet_insurance",
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
    p.add_argument("--claims-scope", action="store_true",
                    help="Phase 4 (plan_claim_answer_correctness.md): report policy_type "
                         "distribution among claims-vocabulary chunks in the multi-topic "
                         "reference docs, instead of the standard minority/disagree audit")
    args = p.parse_args()

    chunks = load_chunks(args.meta_path)

    if args.claims_scope:
        result = audit_claims_scope(chunks)
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print_claims_scope_report(result)
        return 0

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
