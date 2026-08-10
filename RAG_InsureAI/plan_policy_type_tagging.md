# Plan: fix persistent `policy_type` chunk mislabelling

**Status:** diagnosis complete, implementation not started.
**Audience:** implementing agent (Sonnet). Investigation done 2026-07-31 against the live
KB (414 chunks, `app/turbovec_data/documents/insurance_docs_meta.ndjson`).

---

## Why previous fixes didn't hold

Every prior round improved *how well the classifier reasons about an isolated block of
text* — LLM-always instead of regex short-circuit (`project_metadata_classifier_llm_bypass_bug`),
tightened bare-word regex (`project_candidate_vocab_runaway_matching_bug`), a full re-tag
(`project_policy_type_metadata_mistagging`), section-aware chunking
(`project_section_aware_chunking_rebuild`).

None of them addressed the two structural facts below. That is why the symptom keeps
coming back in a new costume.

---

## Root causes (evidence-backed, ranked)

### RC-1 — The section-aware pipeline has never actually run on this corpus
The strongest signal in the codebase — `regex_first_pass_policy_type()`'s **heading-first**
branch (`metadata_tagger.py:1170`) — has never fired for a single chunk in the live KB.

Measured:
```
chunking_method:  414/414 = "semantic"     (never "section")
section_id present:      12/414            (all from Pet_Insurance_Guide.pdf)
section_heading NON-EMPTY: 0/414
```
So `regex_first_pass_policy_type(section_heading="", section_text)` always skips to the
brittle body-keyword branch, and the other 402 chunks never went through that path at all —
they were tagged by `classify_chunk_policy_type()` (text-only).

**A fix was built and then never took effect on the data.** Any plan that doesn't verify
`chunking_method` / `section_heading` actually change on disk will repeat this.

### RC-2 — No document-level topic prior exists anywhere in the pipeline
Both classifiers take text and nothing else:
- `classify_chunk_policy_type(text, llm, *, force_llm)` — `metadata_tagger.py:1055`
- `verify_and_enrich_section_metadata(section_text, assigned_type, llm)` — `metadata_tagger.py:1351`

`classify_document_type()` (`metadata_tagger.py:127`) **is not a topic classifier** — it
returns document *format* (`policy_document` / `reference_handbook` / `regulatory` /
`general`). Nothing tells a chunk "you came from `travelinsuranceguide.pdf`".

Consequence — minority-class contamination inside single-topic documents:

| Source document | Expected | Actual tag distribution |
|---|---|---|
| `5e9acf857576_travelinsuranceguide.pdf` | travel | 20 travel + **1 motor** |
| `ea22bdd3a9bf_m4-5f.pdf` (health module) | health | 16 health + **2 travel** |
| `df12091601c0_m3-f2.pdf` (life module) | life | 11 life + **1 health** |
| `5a1c9b6aaef2_m4-7f.pdf` (liability module) | liability | 8 liability + 3 general + **1 fire** |

(The large law textbook and handbook are legitimately multi-topic — exclude them from this
particular check.)

### RC-3 — Exclusions and cross-references are scored as topic evidence
Every mistag inspected has the same shape: a foreign type appears in a sentence that is
*denying* or *comparing*, and the classifier counts it as subject matter.

- `6d4707c0` — travel guide → tagged **motor**.
  Text: *"Exclusions in luggage insurance … does not cover … motorised vehicles … electric
  vehicles for which **motor liability insurance** is required."*
  The word "motor" exists purely to say travel insurance **excludes** it.
- `d8d4f439` — liability module → tagged **fire**.
  Text: *"…in General Insurance the cover is granted normally for one year and in **Fire
  Insurance** the preamble states…"* — a passing cross-reference. Page footer literally
  reads *"Liability Insurance & Documents in General Insurance"*.
- `406372d2` — life module → tagged **health**.
  Text is the **life** proposal form's health questionnaire. Footer: *"Practice of Life
  Insurance"*.

### RC-4 — Ground-truth page headers/footers are sitting unused inside the chunk text
These PDFs print their topic on every page and it is already in `page_content`:
`"Practice of Life Insurance"`, `"Liability Insurance & Documents in General Insurance"`,
`"Health Insurance"`, `"Travel Insurance Guide page 15"`, `"MODULE - 4 Practice of General
Insurance"`. Currently treated as ordinary prose that *competes with* body mentions
instead of as a label that *dominates* them.

### RC-5 — `policy_type_confidence` / `all_policy_types` on chunks are stale DOCUMENT-level values
Do **not** build any gate on chunk `policy_type_confidence`. Proven: the value is uniform
per source file (it is written by the document-level `tag_document()`,
`metadata_tagger.py:220/298/300`, and leaked onto chunks on older ingests — `rag.py:705`
and `:776` only added `_CHUNK_SKIP_FIELDS` later).

```
conf=0.667 × 256  → exactly the law PDF's chunk count
conf=0.85  × 21   → exactly the travel guide's chunk count
conf=1.0   × 24   → m3-f2 (12) + m4-7f (12)
64 chunks have no confidence field at all
```
The worst mistag (`406372d2`, life→health) carries `confidence=1.0` — because that is the
*document's* score, not the chunk's. A confidence floor would do nothing.

**Useful side effect:** because these are doc-level, `policy_type ∉ all_policy_types` is a
free, deterministic "this chunk disagrees with its own document" detector. **40/414 chunks**
currently trip it. Use it as an audit signal (Phase 0), not as an auto-corrector.

---

## Implementation plan

Ordering matters: Phase 0 first (you cannot tell if this worked without it), then 1→2 which
are high-value and low-risk, then 3→4 which touch ingestion.

### Phase 0 — Measurement harness *(no behaviour change; do this first)*
1. New script `RAG_InsureAI/policy_type_audit.py`:
   - per-source `policy_type` distribution (the table in RC-2);
   - flag chunks whose tag is a minority class inside a single-topic document;
   - flag `policy_type ∉ all_policy_types` (currently 40);
   - exit non-zero above a baseline, so it can become a standing gate like
     `contamination_corpus_runner.py`.
2. **Hand-label a gold set of ~60–80 chunks** stratified across all 10 source documents,
   stored as `RAG_InsureAI/policy_type_gold.json`. Include every chunk named in RC-2/RC-3.
   Without this, "did it improve?" is unanswerable — every previous round failed partly for
   this reason.
3. Record the baseline accuracy number in the file before changing anything.

### Phase 1 — Give both classifiers a document-level topic prior *(highest value)*
1. Add a `derive_document_topic_prior(filename, sample_chunk_texts) -> tuple[str, float]`
   helper in `metadata_tagger.py`. Compute it **once per document** at ingestion:
   - run existing `classify_query_policy_type()` on the *filename* (cheap, and
     `travelinsuranceguide.pdf` resolves immediately);
   - plus a majority vote over `regex_first_pass_policy_type()` across a sample of the
     document's own sections;
   - return `("general", 0.0)` when the document is genuinely multi-topic (the law textbook
     and handbook **must** land here — verify this explicitly).
2. Thread it through as keyword-only params (keeps call sites backward compatible):
   - `classify_chunk_policy_type(..., *, doc_prior: str = "", doc_prior_conf: float = 0.0)` — `:1055`
   - `verify_and_enrich_section_metadata(..., *, doc_prior: str = "")` — `:1351`
3. Inject into both prompts — `_build_policy_type_prompt()` (`:999`) and
   `_build_verify_and_enrich_prompt()` (`:1243`) — with **explicit override permission**, e.g.:
   > This chunk comes from a document that is overall about **{doc_prior}** insurance. Most
   > chunks in it are **{doc_prior}**. Answer a different type ONLY if this chunk is
   > substantively about that other type. A passing mention, a comparison, or an exclusion
   > list is NOT enough to change the answer.

   Omit the block entirely when `doc_prior == "general"`.
4. Update call sites: `rag.py:388-389`, `metadata_tagger.py:1658`
   (`classify_chunk_policy_type_batch`).

**Risk:** over-anchoring — a genuinely multi-topic document could get flattened to one type.
Mitigated by (a) returning `general` for multi-topic docs, (b) explicit override wording,
(c) the Phase 0 gold set must include multi-topic-doc chunks that *should* differ from their
doc prior.

### Phase 2 — Teach both prompts that exclusions/cross-references are not topics
Add a rule + two real few-shot examples (use `6d4707c0` and `d8d4f439` verbatim) to the same
two prompt builders:
- mentions inside exclusion language (*"does not cover"*, *"excluded"*, *"not covered"*,
  *"unless"*) are **evidence against**, never for;
- cross-references (*"as in X insurance"*, *"unlike X"*, *"whereas in X"*) and comparisons
  describe a *different* product;
- the type a chunk *is about* is the one whose rules/benefits/procedures the chunk states.

Cheap, self-contained, and directly targets the confirmed failure shape.

### Phase 3 — Mine the repeated page header/footer as a per-page topic label
1. At ingestion, per document, find lines that repeat across ≥3 pages (running header/
   footer) and extract the topic-bearing one. Store as chunk metadata `page_header_topic`.
2. Pass it as the `section_heading` argument to `regex_first_pass_policy_type()` when the
   real `section_heading` is empty. **This revives the strongest branch, which currently
   never fires** (RC-1).
3. Strip the detected running header from chunk text *before* body-keyword scoring so it
   labels rather than pollutes.

### Phase 4 — Make the section-aware path actually execute
1. Diagnose why `section_heading` is empty for 414/414 and `chunking_method` is never
   `"section"` — inspect `SectionChunker` in `rag.py` and `semantic_chunker.py` against
   these specific PDFs (they are module scans; the heading regex likely expects markdown-ish
   or numbered headings that never match).
2. Either fix the heading extractor or formally accept Phase 3's `page_header_topic` as its
   substitute — but do not leave the codebase claiming a section-aware path that no data
   goes through.
3. Re-ingest and **assert on disk** that `chunking_method` and `section_heading` actually
   changed. This is the check that was missing last time.

### Phase 5 — Re-tag and verify
1. Re-tag the corpus with Phases 1–3 active (back up the `.ndjson` first — a
   `.bak-<epoch>` convention already exists in that directory).
2. Gold-set accuracy must beat the Phase 0 baseline; the RC-2 minority-class contamination
   must drop to ~0 for the four single-topic documents.
3. Run `contamination_corpus_runner.py` — clean/exemption controls must stay at **0%**.
4. Spot-check the two live questions this surfaced from:
   *"Is accidental damage to my phone covered under home insurance?"* and
   *"Does my travel policy cover a cancelled flight?"*

### Phase 6 — Guardrails against silent regression
1. Stop writing document-level `policy_type_confidence` / `all_policy_types` onto chunks, or
   rename them `doc_policy_type_confidence` / `doc_all_policy_types` so nobody builds a
   chunk-level gate on them again (RC-5).
2. Wire `policy_type_audit.py` in as a standing check alongside the contamination runner.

---

## Explicit "do not do" list
- **Do not** gate on chunk `policy_type_confidence` — it is a document-level value (RC-5).
- **Do not** reintroduce a regex short-circuit that skips the LLM — already reverted once
  (`project_metadata_classifier_llm_bypass_bug`).
- **Do not** auto-correct chunks purely because `policy_type ∉ all_policy_types` — that
  compares against *document* evidence; use it to flag for review, not to rewrite.
- **Do not** ship a re-tag without re-reading the `.ndjson` from disk to confirm the values
  actually changed (RC-1 is exactly this failure).

## Out of scope
The `home` insurance content gap — the KB has no real homeowner's document, so
*"is accidental damage to my phone covered under home insurance?"* cannot be answered
correctly by any tagging fix. That is a content-acquisition question, tracked separately.

## Already fixed during this investigation
Two chunks from `5e9acf857576_travelinsuranceguide.pdf` (`e687a829`, `dd789f24`) were
mistagged `policy_type: home` and corrected to `travel` in the live `.ndjson`. They are a
clean instance of RC-2/RC-3 and should be in the Phase 0 gold set as regression anchors.
