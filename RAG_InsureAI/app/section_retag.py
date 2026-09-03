"""
Full-KB `section` re-tag using the EXISTING, already-correct classification
pipeline (classify_chunk_intent() — the same function
classify_chunk_intents_batch()/SectionChunker.split_documents() call at
ingestion time), same two-step classify -> review -> apply workflow as
policy_type_retag.py.

Why this script exists: traced a real "types of life insurance" retrieval
failure (2026-09-01) to several chunks with genuine, correct body content
(e.g. "3) Endowment Assurance", "6) ULIP" in insurance hb 1101.pdf) tagged
`section: general` instead of `types_of_insurance`/`benefits` — metadata-
first retrieval and the section-intent-mismatch filter both key off the
stored `section` tag, so this real content never reached the model, which
then fabricated plausible-sounding definitions from its own training data
instead (correctly caught and dropped by the post-generation faithfulness
checker, which is why the symptom looked like a checker bug at first).
Same shape as the policy_type mistagging bug policy_type_retag.py fixed —
per-chunk classification exists and is accurate, the stored data is just
stale relative to it (older ingestion pass, or chunk never re-synced).

Confirmed live that regex-only classification (llm=None) is NOT reliable
enough to bulk-apply here: "7) Universal Life Insurance" (real content
about a ULIP-hybrid product) regex-scored "premiums" as its top label
purely because its body happens to mention "premium" several times while
explaining the product — same "regex over-matches an incidental keyword"
false-positive risk already documented for the policy_type scan. This
script therefore always classifies WITH the real classification LLM
(classify_chunk_intent's own internal regex-confident gate still skips
the LLM call for genuinely unambiguous chunks — this doesn't force an LLM
call per chunk, just makes one available for the ambiguous ones, exactly
matching what classify_chunk_intents_batch does at real ingestion time).

Two-step workflow, safety, and usage: identical to policy_type_retag.py —
see that script's own docstring. The only differences are the field
(`section` instead of `policy_type`) and the classifier call
(classify_chunk_intent() instead of verify_and_enrich_section_metadata()).

Usage (run INSIDE the api container):
    python3 section_retag.py                # classify + dry-run summary, writes decisions JSON
    python3 section_retag.py --limit 20      # smoke-test on first 20 chunks
    python3 section_retag.py --apply         # replay the saved decisions onto the .ndjson
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "/app/app")

DEFAULT_META_PATH = "/app/app/turbovec_data/documents/insurance_docs_meta.ndjson"
DEFAULT_DECISIONS_PATH = "/app/app/section_retag_decisions.json"


def load_records(meta_path: str) -> list[tuple[str, dict]]:
    records = []
    with open(meta_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            records.append((line, json.loads(line)))
    return records


def cmd_classify(args) -> int:
    from metadata_tagger import classify_chunk_intent
    from rag import RAGPipeline

    pipe = RAGPipeline()
    llm = pipe._get_llm()
    if llm is None:
        print("ERROR: no LLM available (get_classification_llm failed) — refusing to run "
              "regex-only re-tag, that would reintroduce the exact false-positive risk this "
              "fixes (confirmed live: regex-only mistagged a real ULIP chunk as 'premiums').")
        return 1

    all_records = load_records(args.meta_path)
    total = len(all_records)
    records = all_records[: args.limit] if args.limit else all_records

    decisions = []
    changes = []
    t0 = time.time()
    for i, (raw_line, d) in enumerate(records):
        meta = d.get("metadata", {}) or {}
        old_section = meta.get("section", "general")
        text = d.get("text", "") or d.get("page_content", "") or ""
        if not text.strip():
            continue

        src = meta.get("source", "?")
        heading = meta.get("section_heading", "") or ""
        doc_type = meta.get("doc_type", "general") or "general"

        new_section = classify_chunk_intent(text, doc_type=doc_type, llm=llm, heading=heading)

        decisions.append({
            "id": d.get("id"), "source": src, "heading": heading,
            "old": old_section, "new": new_section,
        })
        if new_section != old_section:
            changes.append(decisions[-1])

        if (i + 1) % 25 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t0
            print(f"[{i + 1}/{len(records)}] {elapsed:.0f}s elapsed, "
                  f"{len(changes)} changes so far", file=sys.stderr)

    print(f"\n{len(changes)} chunks would change out of {len(records)} processed "
          f"({total} total in file)")
    by_source: dict[str, int] = {}
    for c in changes:
        by_source[c["source"]] = by_source.get(c["source"], 0) + 1
    for src, n in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {n:>4}  {src}")
    print()
    for c in changes:
        print(f"  {(c['id'] or '')[:8]}  {c['old']:<18} -> {c['new']:<18}  {c['heading'][:40]!r:42}  {c['source'][:40]}")

    with open(args.decisions_path, "w", encoding="utf-8") as fh:
        json.dump({
            "meta_path": args.meta_path,
            "total_chunks_in_file": total,
            "chunks_classified": len(records),
            "limited": bool(args.limit),
            "decisions": decisions,
        }, fh, indent=2)
    print(f"\nWrote {len(decisions)} decision(s) to {args.decisions_path}")
    print("Review the diff above, then run --apply to replay these EXACT decisions "
          "(no re-classification) onto the .ndjson.")
    return 0


def cmd_apply(args) -> int:
    try:
        with open(args.decisions_path, encoding="utf-8") as fh:
            saved = json.load(fh)
    except FileNotFoundError:
        print(f"ERROR: no decisions file at {args.decisions_path} — run classify first "
              f"(plain invocation with no --apply), review its diff, then --apply.")
        return 1

    if saved.get("limited"):
        print("ERROR: the saved decisions were produced with --limit (a smoke-test run) — "
              "refusing to apply a partial classification pass to the full KB. "
              "Re-run classify without --limit first.")
        return 1

    all_records = load_records(args.meta_path)
    total = len(all_records)
    if saved.get("total_chunks_in_file") != total:
        print(f"ERROR: decisions file was produced against a .ndjson with "
              f"{saved.get('total_chunks_in_file')} chunks, but {args.meta_path} currently "
              f"has {total} — the KB changed since classify ran. Re-run classify first.")
        return 1

    # Only apply the ids explicitly approved via --apply-ids (comma-separated
    # id prefixes) if given; otherwise apply every changed decision.
    changes = [d for d in saved["decisions"] if d["old"] != d["new"]]
    if args.apply_ids:
        approved_prefixes = set(args.apply_ids.split(","))
        changes = [c for c in changes if any((c["id"] or "").startswith(p) for p in approved_prefixes)]
    changed_ids = {c["id"] for c in changes}
    change_by_id = {c["id"]: c for c in changes}

    backup_path = args.meta_path + f".bak-{int(time.time())}"
    with open(args.meta_path, encoding="utf-8") as src, open(backup_path, "w", encoding="utf-8") as dst:
        dst.write(src.read())
    print(f"Backed up original to {backup_path}")

    written = 0
    with open(args.meta_path, "w", encoding="utf-8") as fh:
        for raw_line, d in all_records:
            rid = d.get("id")
            if rid in changed_ids:
                d.setdefault("metadata", {})["section"] = change_by_id[rid]["new"]
                fh.write(json.dumps(d) + "\n")
                written += 1
            else:
                fh.write(raw_line + "\n")

    print(f"Applied {written} corrected section value(s) to {args.meta_path}")
    print("Restart the api container to load the corrected metadata.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--meta-path", default=DEFAULT_META_PATH)
    p.add_argument("--decisions-path", default=DEFAULT_DECISIONS_PATH)
    p.add_argument("--apply", action="store_true",
                    help="replay the saved decisions file onto the .ndjson (no LLM calls)")
    p.add_argument("--apply-ids", default="",
                    help="comma-separated id prefixes to apply (default: apply every changed "
                         "decision in the file) — use after manually reviewing the diff to "
                         "apply only the confirmed-correct subset")
    p.add_argument("--limit", type=int, default=0,
                    help="classify only the first N chunks (0 = all); smoke-test only, "
                         "--apply refuses a --limit'd decisions file")
    args = p.parse_args()

    if args.apply:
        return cmd_apply(args)
    return cmd_classify(args)


if __name__ == "__main__":
    sys.exit(main())
