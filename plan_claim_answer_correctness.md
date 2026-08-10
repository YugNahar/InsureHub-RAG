**STATUS: ALL PHASES COMPLETE AND DEPLOYED (2026-08-04).** See per-phase
results inline below. Final verification: `claims_factual_repro` (the 3
confirmed 2026-08-04 bugs) 0/9 after every phase including Phase 6's cleanup;
`claims_cross_type_repro` (12 types x 3 phrasings, auto-generated) 0/36 with
Phase 1 alone; `known_contamination_repro` (pre-existing corpus) 0/24, no
regression. `docker logs` confirmed Phase 1's gate — not luck — is what
catches the case the retired hand-rolled check used to own.

# Plan: fix claim-process answers (correctness + cross-topic contamination)

## Why this plan exists

Every "how do I claim X" query tested in the 2026-08-04 session produced at
least one wrong or contaminated point. Claim-process questions are the
worst-performing query class in this system. Three confirmed failures:

| Query | Bad output | Reality (KB source) |
|---|---|---|
| travel claim | "file a claim on My Pages / via If Mobile / select If Travel Insurance" | competitor **If P&C**'s own portal/app names, verbatim from `5e9acf857576_travelinsuranceguide.pdf` |
| motor claim | "registration certificate of the stolen vehicle must be **transferred in the name of the insured**" | `m4-3f.pdf` p14 says the insurer asks the RTO **NOT to transfer ownership** — an inversion |
| life claim | "must be completed within 30 days **from the date of the incident**"; "if the claim involves a third party, such as a **car accident**" | IRDA Reg. 8(3) = 30 days from **receipt of all papers**; third-party/MACT is **motor**, absent from life context |

All three were patched individually with regexes in
`_fc_line_is_always_false_claim`. That list now has **6 entries** and is pure
whack-a-mole. Per `feedback_generalize_fixes` and
`feedback_prefers_durable_fix_over_manual_maintenance`, this plan replaces the
per-instance patching with structural fixes.

### Root causes

- **A. Claims content is structurally cross-cutting in the KB.** The chapter
  "General Insurance – Practices & Procedures – Focus Claims"
  (`9.3 INSURANCE LAW AND PRACTICE.pdf` ~p190-210) covers claims for *all*
  policy types in one continuous run of pages. Its chunks get individually
  tagged motor/general/health/life, so any claim query retrieves a mixed-type
  set. The `policy_type` pre-filter cannot separate them.
- **B. Claim vocabulary is type-agnostic.** "claim form", "survey report",
  "assessment", "settlement", "documents", "third party" appear in every
  type's claims section, so the reranker cannot discriminate either.
- **C. Detailed mode amplifies it.** "How to claim X" routes to
  `DETAILED_GROUNDED_PROMPT`, which asks for many numbered points. With a thin
  per-type claims section the model pads from adjacent wrong-type chunks.
- **D. `_TYPE_GIVEAWAY_TERMS` is gated off exactly when needed.** Per
  `project_type_agnostic_jargon_false_positive` it only fires when
  `classify_query_policy_type() != general` — and claim queries often classify
  as `general`/ambiguous, disabling the one check that would have caught the
  motor-in-life leak.

### The two failure classes (design the fixes around these, not the instances)

- **Class A — wrong-type / wrong-source import.** A concept correctly belonging
  to another policy type (or another *company*) is asserted in this answer.
  Covers: third-party/car-accident in life, motor jargon in PA, marine/crop in
  motor, and the If P&C brand leak. **Systematically fixable** via type
  attribution (Phases 1-2).
- **Class B — factual inversion / relation distortion.** Real words from a real
  retrieved sentence, recombined into a false relationship. Covers: the
  stolen-vehicle transfer inversion and the 30-day trigger swap. **Not**
  reachable by type attribution — needs relation-level grounding (Phase 3).

---

## Phase 0 — Measure before fixing (do this first, it gates everything)

A prior point-relevance gate was built, then reverted because it showed **zero
benefit at 280-run scale** (`project_point_relevance_ratio_to_max_unsafe`).
Do not repeat that: get a baseline number before writing any gate.

- Extend the existing `contamination_corpus_runner.py` (already has a
  calibrated PASS/FAIL exit code — `project_contamination_regression_command`)
  with a **claims sub-corpus**: for each of the 12 policy types × 3 phrasings
  ("how do I claim X insurance", "what is the X claim process",
  "walk me through claiming on an X policy"), ≥3 repeats each.
- Label every generated point: `correct` / `class-A wrong-type` /
  `class-B factual` / `ungrounded` / `padding`.
- Emit a per-class rate. **This is the acceptance metric for Phases 1-3.**
  A phase that does not move its target class's rate gets reverted, not kept.

---

## Phase 1 — Per-point type-attribution gate (primary Class A fix)

**File:** `RAG_InsureAI/app/multi_source_rag.py`, inside `ask_stream`, as a new
block alongside the existing per-unit checks (near `_artifact_mismatched`,
~line 8478 — verify, line numbers shift).

> **Correction from the original plan (found during Phase 4, 2026-08-04):**
> step 3 originally said to consult `all_policy_types` to tell "this chunk
> legitimately covers many types" apart from "wrong-type import." Verified
> live against the real metadata file (`insurance_docs_meta.ndjson`):
> `all_policy_types` is **document-level, not per-chunk** — every one of the
> 256 chunks in `9.3 INSURANCE LAW AND PRACTICE.pdf` carries the exact same
> value (`"life, motor"`), regardless of whether that individual chunk is
> actually about marine, health, personal_accident, fire, or anything else
> (confirmed: an individually-`marine`-tagged chunk, an individually-
> `personal_accident`-tagged chunk, etc. all still show `all=[life, motor]`).
> Every other source document in the KB shows the same one-value-per-document
> pattern. This is the same class of bug already documented for
> `policy_type_confidence` in `policy_type_audit.py`'s own docstring (a
> document-level value leaking onto every chunk). **Do not use
> `all_policy_types` for per-chunk disambiguation in Phase 1 — it cannot
> distinguish chunks within a multi-topic document.** Use each retrieved
> chunk's own `policy_type` tag instead (see step 3 below).

For each generated unit (numbered point, else sentence):

1. Detect type-specific concepts using `_TYPE_GIVEAWAY_TERMS` **plus** the
   open-vocab candidate types from `candidate_vocab.py` / `get_active_vocab()`.
2. Resolve the unit's implied type(s).
3. Drop the unit when its implied type ≠ the query's type **and** no chunk
   *actually retrieved for this answer* has `policy_type` (or, per-chunk,
   `candidate_policy_type`) equal to that implied type — check the retrieved
   chunks' individual tags directly (same principle already used correctly by
   the `_TP_MOTOR_ACCIDENT_RE` grounding check added to
   `_fc_line_is_always_false_claim` today: check `_ctx_norm_joined`, the
   actual retrieved text, never a metadata field that turned out to be
   document-level).

**Critical guards** (each closes a known past regression):

- **Skip the whole gate when `classify_query_policy_type()` returns `general`
  or low confidence.** A general query legitimately spans types
  (`project_type_agnostic_jargon_false_positive`).
- Reuse the `_ctx_norm_joined` / `_normalize_word_forms` normalisation so
  plural/possessive forms match, and the de-wrapped context (`_dewrapped_ctx`,
  ~line 8368) so PDF mid-sentence line wraps can't fake an absence.
- Bare single words are the historic false-positive source
  (`project_motor_bareword_giveaway_generalization`,
  `project_compound_word_joiner_bug`) — require a multi-word phrase or a
  two-hit threshold before attributing a type, never one bare noun.
- If **every** unit is dropped, fall through to the existing
  refusal + escalation path rather than emitting a fragment.
- Reuse the established unit-split → drop → renumber →
  `\n{3,}`→`\n\n` collapse pattern verbatim. Do **not** hand-roll a new one.

This is *not* the reverted Phase-2 gate: that used a noisy semantic
similarity ratio-to-max. This is a discrete, explainable lookup against
curated vocabulary.

---

## Phase 2 — Claim-scoped retrieval (secondary Class A fix)

**Files:** `multi_source_rag.py` (retrieval + reranking), `metadata_tagger.py`.

- Add a claim-intent detector: `\b(claim|claiming|settle\w*|reimburs\w*)\b`
  plus procedural shape (how/what is the process/steps/procedure). Keep it in
  one named constant so Phase 6's tests can import it.
- When claim-intent **and** a confident specific query type:
  - Strengthen the existing 0.5× general-tagged down-weight
    (`project_general_chunk_reranking_downweight`) to ~0.3× **for claim
    queries only** — general-tagged claims chunks are exactly the vector for
    cross-type import.
  - Widen `top_k` for the *matching* type so the answer isn't starved into
    padding (`project_topk_reranker_candidate_pool` has the prior calibration).
- Do not hard-filter general-tagged chunks out entirely — `project_metadata_retrieval_filter`
  and the `_retrieved_specific_types` fix showed that starves legitimate answers.

**IMPLEMENTED (2026-08-04):**
- `_is_claim_intent_query()` added (`multi_source_rag.py` ~1391-1408) — vocab
  regex AND procedural-shape regex, matches all 3 of Phase 0's canonical
  phrasings plus common variants ("explain how...", "tell me...").
- `_TYPE_MISMATCH_DISCOUNT` (~5833) now conditional: `0.3` when
  `_is_claim_intent_query(retrieval_query) and _query_policy_type != "general"`,
  else the original `0.5`. Comment cross-references the Phase 4 audit finding
  (29/50 general-tagged claims chunks) as the rationale.
- **`top_k` widening deliberately DEFERRED, not forgotten.** `_doc_top_k`/
  `_chunk_limit` are computed (~5396/5412) *before* `_query_policy_type`
  exists (~5467) — widening would need restructuring that ordering. More
  importantly: per `plan_shallow_answers_context_budget.md`, the LLM is
  currently shown a **900-character** context budget regardless of how many
  chunks are retrieved (every chunk gets fragmented into ~112-char slivers
  once more than ~8 compete for that budget). Widening `top_k` before that
  budget is fixed would add MORE competing chunks to an already-starved
  budget and could make fragmentation worse, not better. Revisit top_k
  widening **after** the other plan's Phase C lands, not before.

---

## Phase 3 — Relation-anchored grounding (Class B fix)

**File:** `multi_source_rag.py`, per-unit checks, reusing `_source_window`
(~line 8491) which already scopes a match to its surrounding sentence.

Generalise the two hand-patched Class B bugs into one mechanism. For a unit
containing either shape:

- **temporal deadline** — `within N days/months of <ANCHOR>`
- **directional transfer/assignment** — `<OBJ> transferred/assigned/paid to <ANCHOR>`

extract the `(relation, anchor)` pair and require **both the relation phrase
and the anchor to co-occur inside one `_source_window` of the de-wrapped
context**. Drop the unit when they don't.

This is the generalisation of: "30 days" grounded but re-anchored to *incident*
instead of *receipt of papers*; "transfer" grounded but re-anchored to *the
insured* instead of *nobody*. Word-level grounding passes both; relation-level
grounding catches both.

Start **log-only** (count would-drops against the Phase 0 corpus) before
enabling, exactly as the contamination gate was rolled out.

---

## Phase 4 — Audit the claims-chapter metadata

**Files:** `RAG_InsureAI/policy_type_audit.py`, `RAG_InsureAI/app/policy_type_retag.py`.

Root cause A may be partly a *tagging* problem, not only a structural one.
Scope an audit to chunks whose `section` is claims-like or whose text carries
claims vocabulary, and report the `policy_type` / `all_policy_types`
distribution across `9.3 INSURANCE LAW AND PRACTICE.pdf` p190-210.

Decide from the data (do not assume): a chunk that genuinely covers all types
should stay `general` with `all_policy_types` populated, so Phase 1 can consult
`all_policy_types` rather than treating it as wrong-type. A chunk that is
really motor-only but tagged `general` should be retagged.

Note: task #106 (full 414-chunk retag classify pass, no apply) is still pending
and overlaps — run the claims-scoped audit first, it is cheaper and targeted.

**RESULTS (run 2026-08-04, `policy_type_audit.py --claims-scope`, new mode
added this session):** Claims-vocabulary chunks in the two multi-topic
reference docs are covered by `_CLAIMS_VOCAB_RE` (claim form, survey report,
settlement, third-party claims, no-claim-bonus, loss assessor, discharge
voucher, condition of average, etc.) and `_looks_like_toc()` (excludes
table-of-contents chunks — an unfiltered first pass falsely suggested "58% of
claims content is general-tagged" purely from ToC page-number lines like
"Claims Procedure in Respect of a Life Insurance Policy … 232").

- 50 genuine claims-body chunks found (+ 3 ToC chunks correctly excluded).
- `policy_type` distribution: 29 `general`, 7 `life`, 6 `motor`, 3 `fire`,
  2 `marine`, 2 `health`, 1 `personal_accident`.
- **`all_policy_types` is confirmed document-level, not per-chunk** — see the
  correction added to Phase 1 above. This was the actual reason every sampled
  chunk showed `all=[life, motor]` regardless of its real individual topic;
  it is not usable for Phase 1's disambiguation as originally written.
- 29/50 (58%) of the genuine claims-body chunks are tagged `general` — this
  is real, not a ToC artifact, and is itself root cause A's direct evidence:
  the claims chapters mix all types' claims content inside `general`-tagged
  narrative prose (procedural steps, principles, licensing) that individual
  chunking didn't separate by type. This is the population Phase 2's
  general-tagged down-weight (currently 0.5x, plan proposes 0.3x for claim
  queries) is designed to act on.
- Retag decision: do NOT bulk-retag these 29 `general` chunks — spot-checking
  the samples (subrogation principles, licensing/broker fee schedules, IRDA
  claims-procedure regulations, surveyor/loss-assessor qualification rules)
  shows they are genuinely type-agnostic regulatory/procedural content, not
  mistagged single-type content. The fix belongs in retrieval weighting
  (Phase 2) and per-point attribution (Phase 1), not re-tagging.

---

## Phase 5 — Prompt rules (belt-and-braces only, never the primary fix)

**File:** `RAG_InsureAI/app/prompt_template.py` — all **three** prompts
(`CONVERSATIONAL_RAG_PROMPT`, `STRICT_GROUNDED_PROMPT`,
`DETAILED_GROUNDED_PROMPT`); they share no rule inheritance, so each needs the
rule added separately.

Add: when answering a claim question about one insurance type, never import a
procedure, document, or body specific to a different type (e.g. no third-party
motor-accident or MACT steps in a life/health/travel claim answer).

Explicitly secondary: `project_dont_trust_buried_disclosure_instructions`
established the small vLLM model ignores prompt instructions under pressure.
Phases 1-3 must stand on their own without this.

---

## Phase 6 — Retire the whack-a-mole regexes + lock in the gate

Once Phases 1-3 pass Phase 0's corpus:

- The **Class A** entries in `_fc_line_is_always_false_claim` become redundant —
  the stolen-vehicle-transfer trio and the third-party/motor-accident check.
  Remove them **only if** the corpus shows the new gates catch those cases;
  re-run to prove no regression.
- **Keep** the genuine always-false entries (TPA expansion, fines coverage,
  condition-of-average, 30-day trigger, premium/market-value timing) — those
  are factual errors, not type errors, and the new gates do not subsume them.
- Update the shared log line if entries are removed (it now enumerates all six).
- Add the claims sub-corpus to the standing regression command so this class
  can't silently regress.

---

## Testing discipline (hard-won this session — do not skip)

1. **KV cache invalidates repeat testing.** Identical query text returns the
   cached answer *without re-running the correction pipeline* — a fix will look
   broken or look fine for the wrong reason. Always use a fresh phrasing, and
   confirm in `docker logs` you see **`KV cache stored`**, not `KV cache hit`.
   `rm -f query_kv_cache.json` does NOT bust it (in-memory) —
   `project_kv_cache_breaks_repeat_testing`.
2. **Never judge from raw streamed tokens.** The live SSE stream is the
   *pre-correction* text and always shows the bug. The fix only appears in the
   trailing JSON's `corrected_text`. Use `--max-time 120` minimum and parse that
   field — a short `--max-time` truncates before it arrives (this produced a
   completely false "the fix failed" reading earlier today).
3. **Deploy with `docker compose up -d --force-recreate --no-deps api`.**
   `docker restart` does not reload `.env` or reliably pick up code.
4. **Refusal-path queries send real escalation emails** to five agent addresses
   (`project_test_queries_trigger_real_emails`) — expect them when a whole-reply
   discard fires during corpus runs.
5. Verify every symbol/line number in this plan before editing — this file
   records them as of 2026-08-04 and `multi_source_rag.py` shifts constantly.

## Critical files

- `RAG_InsureAI/app/multi_source_rag.py` — `ask_stream`; per-unit checks
  (`_artifact_mismatched` ~8478, `_source_window` ~8491, `_ctx_norm_joined`
  ~8476, `_dewrapped_ctx` ~8368), `_fc_line_is_always_false_claim` ~9477,
  retrieval + reranking
- `RAG_InsureAI/app/metadata_tagger.py` — `classify_query_policy_type`,
  `_TYPE_GIVEAWAY_TERMS`, `get_active_vocab`
- `RAG_InsureAI/app/candidate_vocab.py` — open-vocab types
- `RAG_InsureAI/app/prompt_template.py` — the three prompts
- `contamination_corpus_runner.py` — regression gate to extend
- `RAG_InsureAI/policy_type_audit.py`, `RAG_InsureAI/app/policy_type_retag.py`
