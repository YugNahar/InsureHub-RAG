"""
Full-KB policy_type re-tag using the EXISTING, already-correct classification
pipeline (regex_first_pass_policy_type + verify_and_enrich_section_metadata —
the same two functions SectionChunker.split_documents() calls at ingestion
time), now with a document-level topic prior (derive_document_topic_prior)
so a chunk isn't classified with zero visibility into what document it lives
in.

Why this script exists instead of re-uploading every document: confirmed
live (2026-07-31, see plan_policy_type_tagging.md) that the classifier
code, run fresh against known-mistagged chunks, already gets them right —
the logic isn't broken, the DATA is stale relative to it. Each document was
tagged once at its own upload time, by whatever version of this pipeline
existed then, and nothing has re-synced older chunks since. This script
re-runs the CURRENT pipeline against every chunk's already-stored text and
overwrites policy_type with the fresh result — no re-chunking, no
re-embedding, no vector-index change, since the text and its vector are
unaffected by a tag correction.

Why the doc-prior matters here specifically: a first version of this script
(no prior) was run as a dry-run and manually reviewed. It correctly fixed
5/5 known mistags, but ALSO flipped several genuinely correct travel-guide
chunks to "health" (a travel-medical benefit for "acute toothache during
the journey", read with zero document context, looks like generic health
content) — a real regression, not just noise. See derive_document_topic_
prior()'s own docstring in metadata_tagger.py for the full writeup.

Chunks are classified individually (not grouped into cross-document
sections) because live inspection confirmed section grouping barely
applies to this corpus already — most chunks have no section_heading and
fall back to one-chunk-per-section. The document-level PRIOR is computed
once per source from ALL of that source's own chunk texts, then applied
per chunk — this is deliberately different from re-grouping chunks into
sections; it changes the CONTEXT a chunk is classified with, not the
GRANULARITY of classification (each chunk still gets its own verdict, and
can still land on a genuinely different type than its document's prior
when its own text substantively supports that — see the prompt's explicit
override wording).

Two-step workflow (dry-run and apply are DELIBERATELY separate scripts of
work, not just a flag toggling on the same live pass): --apply must review
the EXACT decisions a human reviewed, not re-run the LLM a second time and
risk shipping a different diff than the one approved. Classification always
writes its decisions to --decisions-path; --apply reads that file back
rather than re-classifying.

Safety:
- Writes a timestamped backup of the .ndjson before any change (same
  convention already used in this data directory: `.bak-<epoch>`).
- --apply refuses to run unless a decisions file from THIS SAME classify
  pass exists and its chunk count matches what's currently in the .ndjson
  (a stale decisions file from a different KB state must not be replayed).

Usage (run INSIDE the api container, where metadata_tagger/rag are importable):
    python3 policy_type_retag.py                        # classify + dry-run summary, writes decisions JSON
    python3 policy_type_retag.py --limit 20              # smoke-test on first 20 chunks
    python3 policy_type_retag.py --apply                 # replay the saved decisions onto the .ndjson
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, "/app/app")

DEFAULT_META_PATH = "/app/app/turbovec_data/documents/insurance_docs_meta.ndjson"
DEFAULT_DECISIONS_PATH = "/app/app/policy_type_retag_decisions.json"


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
    from metadata_tagger import (
        regex_first_pass_policy_type,
        verify_and_enrich_section_metadata,
        derive_document_topic_prior,
    )
    from rag import RAGPipeline

    pipe = RAGPipeline()
    llm = pipe._get_llm()
    if llm is None:
        print("ERROR: no LLM available (get_classification_llm failed) — refusing to run "
              "regex-only re-tag, that would reintroduce the exact bug this fixes.")
        return 1

    all_records = load_records(args.meta_path)
    total = len(all_records)
    # Only the first N get RE-CLASSIFIED when --limit is set (a smoke-test
    # knob) — the doc_prior below is still computed from EVERY chunk of a
    # source's FULL text regardless of --limit, so a limited smoke-test run
    # never sees a truncated, misleading prior.
    records = all_records[: args.limit] if args.limit else all_records

    # Document-level topic prior, computed once per source from ALL of
    # that source's own chunk texts in the file (not just the possibly
    # --limit-sliced sample being classified this run).
    texts_by_source: dict[str, list[str]] = {}
    for _, d in all_records:
        src = (d.get("metadata", {}) or {}).get("source", "?")
        texts_by_source.setdefault(src, []).append(d.get("text", "") or "")
    prior_by_source: dict[str, tuple[str, float]] = {
        src: derive_document_topic_prior(texts, filename=src) for src, texts in texts_by_source.items()
    }
    print("Document-level topic priors:", file=sys.stderr)
    for src, (prior, share) in sorted(prior_by_source.items()):
        print(f"  {prior:<12} (share={share:.2f})  {src}", file=sys.stderr)
    print(file=sys.stderr)

    decisions = []  # every chunk processed, old + new, whether changed or not
    changes = []
    t0 = time.time()
    for i, (raw_line, d) in enumerate(records):
        meta = d.get("metadata", {}) or {}
        old_type = meta.get("policy_type", "general")
        text = d.get("text", "") or ""
        if not text.strip():
            continue

        src = meta.get("source", "?")
        doc_prior, _share = prior_by_source.get(src, ("general", 0.0))

        first_pass = regex_first_pass_policy_type("", text)
        # Anchor-fix (same as rag.py's SectionChunker and api.py's
        # _reclassify_chunks_with_llm): a razor-thin per-chunk regex guess
        # anchors the LLM VERIFY question on the wrong type; start from
        # doc_prior instead whenever it's confident. This script was the
        # one remaining caller of verify_and_enrich_section_metadata still
        # passing the raw first_pass guess as assigned_type — found via
        # `grep -rn "doc_prior if doc_prior"` across the repo after the
        # full-corpus retag kept reproducing the same 7 travel-guide
        # misclassifications despite the fix supposedly being live.
        assigned_type = doc_prior if doc_prior and doc_prior != "general" else first_pass
        enriched = verify_and_enrich_section_metadata(
            text, assigned_type, llm=llm, doc_prior=doc_prior,
        )
        new_type = enriched["policy_type"]

        decisions.append({"id": d.get("id"), "source": src, "old": old_type, "new": new_type})
        if new_type != old_type:
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
        print(f"  {c['id'][:8]}  {c['old']:<18} -> {c['new']:<18}  {c['source'][:40]}")

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

    changes = [d for d in saved["decisions"] if d["old"] != d["new"]]
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
                d.setdefault("metadata", {})["policy_type"] = change_by_id[rid]["new"]
                fh.write(json.dumps(d) + "\n")
                written += 1
            else:
                fh.write(raw_line + "\n")

    print(f"Applied {written} corrected policy_type value(s) to {args.meta_path}")
    print("Restart the api container to load the corrected metadata.")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--meta-path", default=DEFAULT_META_PATH)
    p.add_argument("--decisions-path", default=DEFAULT_DECISIONS_PATH)
    p.add_argument("--apply", action="store_true",
                    help="replay the saved decisions file onto the .ndjson (no LLM calls)")
    p.add_argument("--limit", type=int, default=0,
                    help="classify only the first N chunks (0 = all); smoke-test only, "
                         "--apply refuses a --limit'd decisions file")
    args = p.parse_args()

    if args.apply:
        return cmd_apply(args)
    return cmd_classify(args)


if __name__ == "__main__":
    sys.exit(main())
