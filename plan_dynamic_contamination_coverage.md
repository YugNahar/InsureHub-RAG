# Scope — dynamic, KB-derived cross-topic contamination coverage

**Status (updated 2026-07-29):** **D1 done. D2 measured and FAILED its own kill
criterion — probe reverted from the request path.** D0, D3, D4 not started.
`type_vocab_miner.py` stays in the tree (harmless offline, still usable for future
work) but nothing in `multi_source_rag.py` calls it anymore. Read §6.7 before
attempting this again — the failure mode is understood and a fix direction exists,
just not yet built.

**Problem statement (from the user, and it is the correct framing):** the current
contamination defense is (a) a **hand-maintained list** covering 6 of 12 policy types, and
(b) gated on the offending term **already being present in the retrieved chunks** — but
contamination does not require that. The `discharge summaries` case proved it: the phrase
appears in **zero** KB chunks, so it came from the model's parametric knowledge, and
`_text_has_giveaway_contamination` structurally could not fire.

So the target is a mechanism that is **dynamic** (derived from the KB, no manual list) and
**retrieval-independent** (does not require the term to have been retrieved).

---

## 1. What exists today, and exactly where it stops

| Mechanism | Where | Limitation |
|---|---|---|
| `_TYPE_GIVEAWAY_TERMS` | `multi_source_rag.py:1298` | Hand-written. Covers `crop, fidelity, health, marine, motor, transit`. **`fidelity` and `transit` aren't even KB types.** |
| `_text_has_giveaway_contamination` | `:1330` | `if not any(term in context_lower ...): continue` — **requires the term in retrieved context**. Blind to parametric hallucination by construction. |
| `_ARTIFACT_NOUNS` (Fix A, today) | `:1340` | Retrieval-independent ✅ but still a **closed 18-item list**. |
| `_point_grounded` | `:7047` | Per-word overlap; threshold `min(4, ceil(n/2))` → long points pass at ~20% grounding. |
| Phase 2 semantic gate | — | **Built, 280 runs, zero benefit, reverted.** Do not naively rebuild. |

**Measured coverage gap.** KB `policy_type` values actually present (12):
`commercial, crop, fire, general, health, home, liability, life, marine, motor,
personal_accident, travel`.
Protected by a giveaway list (4 of those): `crop, health, marine, motor`.
**Unprotected (8): `commercial, fire, general, home, liability, life, personal_accident,
travel`.**

---

## 2. Existing machinery this should build on — do not start from scratch

This repo already solved "derive vocabulary from the KB at runtime" **twice**. Reuse it.

- `candidate_vocab.py` — `get_active_vocab_extra()`, `promote_to_active_vocab()`;
  persistence + locking already handled.
- `metadata_tagger.py:959` — `get_active_vocab()`; `:1546` `classify_candidate_type()`.
- `_ANCHOR_TYPE_RE` — already auto-extends from KB text at runtime
  (`project_dynamic_anchor_type_discovery`), precedent for exactly this pattern.
- `classify_query_policy_type()` — already gives the query's own type, already computed
  once per request; the contamination check already consumes it.

**Known trap to inherit deliberately:** `project_candidate_vocab_runaway_matching_bug` —
a 1-keyword cheap match with no threshold mislabeled 202/203 unrelated chunks identically.
Any distinctiveness scoring below must carry a **minimum-hit threshold** and must **not**
grow its own vocabulary from weak matches.

---

## 3. Proposed mechanism — per-type distinctive vocabulary, mined from the KB

### 3.1 Offline/startup: build the term→type distinctiveness map

For every `policy_type` in the KB, compute terms that are **strongly associated with that
type and rare in every other type** — a TF-IDF / pointwise-mutual-information style score
over the chunk corpus, grouped by `policy_type`.

- Unit: unigrams **and** bigrams (the real giveaways are phrases — "domiciliary
  hospitalization", "bill of lading", "discharge voucher").
- Keep a term for type *T* only if it clears **both** an absolute frequency floor in *T*
  (not a one-off typo/OCR artifact) **and** a distinctiveness ratio vs. all other types.
- Exclude terms that appear in `general`-tagged chunks above a threshold — generic
  insurance vocabulary ("premium", "policyholder", "claim") must never become a giveaway.
- Persist to disk beside the existing vocab artifacts; rebuild on ingestion (there is
  already a background re-classification hook on upload to piggyback on).

**This is the dynamic part.** New documents → new types and new jargon are picked up with
no code change, which is what the user asked for.

### 3.2 Request time: score the ANSWER, not the retrieved context

For a query classified as type *Q*:
1. Extract candidate terms from the generated answer (same normalization as Fix A —
   reuse `_normalize_word_forms`).
2. Look each up in the distinctiveness map.
3. Flag a unit if it contains a term whose distinctive type *T* ≠ *Q*.

**Critically: this does NOT consult retrieved context**, which is exactly what makes it
catch the `discharge summaries` class. It asks "is this vocabulary characteristic of a
different product?" — a question that has an answer whether or not the term was retrieved.

### 3.3 Guardrails — these are what make it safe, and Phase 2 is why

Phase 2 failed as a *semantic* gate. This is a *lexical* one with a much narrower claim,
but the same failure modes apply. Non-negotiable:

- **Reuse `_TYPE_QUERY_EXEMPT_WORDS` and `classify_query_policy_type()`** — a query that
  names the other type (comparisons: "difference between fire and marine insurance") must
  be exempt. This is already-solved logic, do not re-derive it.
- **Reuse `_EXCLUSION_LANGUAGE_RE`** — "not covered here, covered under your motor policy"
  is legitimate insurance prose, not contamination. Already exists at `:1323`.
- **Never drop the last surviving unit** (existing guardrail).
- **Minimum 2 distinct giveaway hits** before acting on a unit, per the runaway-matching
  precedent — never fire on a single weak term.
- **Fail open** everywhere.

### 3.4 Migration path for `_TYPE_GIVEAWAY_TERMS`

Do **not** delete it. Run the derived map **alongside** it; the hand list becomes a
high-precision floor that the derived map extends. Retiring the hand list is a separate
decision after the derived map has demonstrably matched it on the 4 overlapping types
(this is the same precondition Phase 3 failed and was correctly skipped over).

---

## 4. Sequencing — measurement first, gate last

The single biggest lesson available here is Phase 2's: **a plausible mechanism shipped
without a measured benefit cost 280 runs and was reverted.** So:

- **D0 — extend the corpus first.** Add repro + clean-control cases for the 8 unprotected
  types. Without this there is literally no instrument that can detect success or
  regression on them, and the current corpus's closed `forbidden_phrases` design can only
  catch vocabulary already observed. **This step has standalone value even if the rest is
  never built** and should be done regardless.
- **D1 — build the distinctiveness map offline.** Ship nothing into the request path.
  Deliverable: the map itself + a human review of the top ~30 terms per type. If those
  terms don't look like real giveaways to a human reading them, stop here — the signal
  isn't there and nothing downstream will rescue it.
- **D2 — wire it in LOG-ONLY.** Emit what it *would* have dropped, via the existing
  contamination trace. Run the full corpus. Measure precision against the labeled cases.
  **Explicit kill criterion: if it flags any clean-control case, or its would-drop
  precision on the repro set is below the hand list's, stop and revert.**
- **D3 — enable as a gate**, only if D2's numbers justify it, thresholds calibrated from
  D2 data rather than guessed.
- **D4 — separately**, revisit `_point_grounded`'s `min(4, ceil(n/2))` threshold with its
  own before/after sweep. Independent of D1-D3; do not bundle.

---

## 5. Honest risk assessment

- **Highest risk: false positives on legitimate cross-references.** Insurance prose
  routinely names sibling products. §3.3's exemptions are the mitigation, but this is the
  thing most likely to sink it — hence log-only first.
- **OCR/extraction noise** in this KB (`project_pdf_text_extraction_corruption`) will
  produce junk "distinctive" terms. The frequency floor is the mitigation; the human
  review in D1 is the check.
- **`general`-tagged chunks are 1 of the 12 types and are genuinely cross-topic** — they
  must be excluded from mining, or every generic term becomes a giveaway.
- **This does not solve fabricated *claims*** (Fix B/C's class) — only wrong-topic
  *vocabulary*. A grammatically fluent, on-topic, factually wrong sentence remains
  uncaught by any mechanism here, and that limitation should be stated plainly rather than
  implied away.

**Realistic outcome:** this should meaningfully raise coverage on the 8 unprotected types
and close the retrieval-independence gap. It will not make contamination zero, and the
corpus baseline has never measured a true zero. Anyone reading this later should not treat
D3 shipping as "contamination solved."

---

## 6. HANDOFF — what already exists, and the exact next step

Written for whoever picks this up (including a different model). **Do not re-implement
§3 — it is built.** Verify against the code before changing anything.

### 6.1 What exists

| Thing | Where | State |
|---|---|---|
| Miner + runtime probe | `RAG_InsureAI/app/type_vocab_miner.py` (new file) | done |
| Persisted vocab map | `RAG_InsureAI/app/turbovec_data/type_vocab.json` | built, regenerate with `write_map()` |
| Log-only probe call | `multi_source_rag.py`, `[vocab_probe]` block just above `_retrieval_contamination_detected = False` | wired, **drops nothing** |
| Import | `multi_source_rag.py`: `import type_vocab_miner as _type_vocab_miner` | done |

Key functions in `type_vocab_miner.py`:
- `load_chunks()` → `[(policy_type, text)]` from the docs ndjson.
- `mine(rows)` → per-type scored terms (document-frequency distinctiveness).
- `build_map()` / `write_map()` → `{type: [term,...]}`, applying the viability floor.
- `foreign_type_hits(text, query_policy_type)` → `{other_type: [terms]}`. **Reporting
  only** — the caller applies exemptions.

Rebuild the map after ingesting new documents:
```bash
docker exec -w /app/app insurehub_api python3 -c "import type_vocab_miner as m; m.write_map()"
```

### 6.2 Calibrated constants — do NOT re-guess these

`MIN_DISTINCTIVENESS = 15.0` was calibrated against a labeled set (21 known-good
giveaways from the validated hand list, 8 known-junk terms observed in the first D1
output). Full table is in the constant's own comment. Summary: 8 → 21/21 good but 8/8
junk (no filtering); **15 → 21/21 good, 6/8 junk (chosen)**; 20 → starts losing good
terms; 50 → 14/21 good. 15 is the knee. Changing it requires redoing that measurement.

`MIN_TERMS_FOR_TYPE = 10` is a **threshold, not a list** — that is the "dynamic" property
the user asked for. Types cross into the map on their own as KB content grows.
`MIN_FOREIGN_HITS = 2` exists because a 1-hit trigger already caused a real incident
(`project_candidate_vocab_runaway_matching_bug`, 202/203 chunks mislabelled).

### 6.3 Measured state as of handoff

Terms per type at threshold 15 (need ≥10 to enter the map):
`life 107, travel 82, motor 72, health 67, liability 12` → **in map**;
`marine 9, crop 9, home 1, fire 0, personal_accident 0, commercial 0` → below floor.

Coverage is the **union** of the hand list and the derived map — the hand list is NOT
being replaced, which is why `marine`/`crop` dropping below the floor costs nothing:
- hand list: crop, health, marine, motor
- derived map adds: **liability, life, travel**
- union: **7 of 12 types** (was 4)
- still uncovered: `fire`, `home`, `commercial`, `personal_accident` (KB-content problem,
  not solvable in code — see §5), plus `general` by design.

Probe behaviour verified offline on hand-written text:
- motor answer containing health vocabulary → flags `health`
  (`chronic, domiciliary, hereditary, nursing, nursing home`)
- clean motor / clean health / clean travel answers → silent.

### 6.4 THE NEXT STEP (this is what "finishing D2" means)

The probe has never run against the labeled corpus. Do this:

1. **Deploy** — the probe is in the working tree; confirm it is in the container
   (`docker cp` both files, `py_compile`, restart). Blocked at time of writing only
   because a corpus run was in flight and a restart would corrupt it.
2. **Disable the query cache first** — `DISABLE_QUERY_CACHE=1` in `RAG_InsureAI/.env`,
   recreate. Non-negotiable: see [[project_kv_cache_breaks_repeat_testing]]; deleting the
   cache JSON does NOT bust it, and `contamination_corpus_runner.py` now aborts if the
   cache is on.
3. **Run** `python3 contamination_corpus_runner.py --repeats 3`, nothing else concurrent.
4. **Extract the probe's verdicts**:
   `docker logs insurehub_api --since 90m 2>&1 | grep "\[vocab_probe\] would-flag"`
5. **Score precision** against the corpus labels: for each would-flag, was that case
   actually labelled contaminated (`known_contamination_repro`) or was it a
   `clean_control` / `exemption_control`?

### 6.5 KILL CRITERIA — apply honestly

Stop and revert (delete the probe block; the miner can stay, it is harmless offline) if
**either** holds:
- it would-flags **any** `clean_control` or `exemption_control` case, **or**
- its would-drop precision on the repro set is **below the existing hand list's**.

Only if it clears both, proceed to D3 (make it gate), with thresholds re-derived from the
D2 numbers rather than reused from §6.2 — the calibration above is a *term-quality*
calibration, not a *production-precision* one.

**Do not skip straight to D3 because the offline examples looked good.** Phase 2 of
`plan.md` did exactly that reasoning and cost 280 runs before being reverted.

### 6.6 Not done, deliberately

- **D0** (corpus cases for the unprotected types) — still unstarted, still worth doing on
  its own merits: without it there is no instrument that can detect regressions on those
  types at all.
- **D4** (`_point_grounded`'s `min(4, ceil(n/2))` threshold) — untouched on purpose;
  independent change, needs its own before/after sweep, must not be bundled.

### 6.7 D2 result: FAIL, reverted (2026-07-29)

`contamination_corpus_runner.py --repeats 3` (56 cases, cache disabled, nothing
concurrent) produced 4 `[vocab_probe] would-flag` events. Two of them were on
**`pet-insurance-direct-01`, a `clean_control` case** (2 of its 3 repeats) — hits
included `'pet insurance'`, `"pet's"`, `'pets'`, `'veterinary'`, all from an answer that
opens *"Sure, let's dive into the world of pet insurance! 1. What Is Pet
Insurance?..."* — i.e. the answer's own, entirely correct topic, flagged as
**health**-foreign vocabulary.

**Root cause:** `pet_insurance` is an open-vocab candidate type
([[project_open_vocab_promotion_wired]]) with no `policy_type` bucket of its own in
`insurance_docs_meta.ndjson` — the miner only ever grouped by the 12 coarse hand-tagged
buckets (§1's table), so pet-insurance content living inside `health`-tagged chunks
donated its vocabulary to health's distinctive-term list. The offline D1 calibration
(21 known-good / 8 known-junk terms) never exercised an open-vocab-candidate query, so
it looked clean and wasn't — precisely the scenario §6.5's warning names: **do not skip
straight to D3 because the offline examples looked good.**

Per §6.5, this is the kill criterion firing exactly as specified — stop and revert,
regardless of how clean the rest looked. It was applied: `foreign_type_hits()`'s call
site was removed from `ask_stream`, and the now-unused `import type_vocab_miner` in
`multi_source_rag.py` was removed too. The miner module itself is untouched and still
runnable standalone; the persisted map and calibrated constants (§6.2) remain valid on
their own terms, they just aren't wired to anything live.

The other two would-flag events (of the 4 total) are informational, not part of the
kill-criterion verdict: one landed on `personal-accident-full-picture-01`
(`known_contamination_repro` — a plausible true-positive candidate, foreign_type=travel,
hits=`['sudden','unexpected']`); one landed on non-corpus traffic that happened to share
the capture window (a home-insurance water-damage answer flagged foreign_type=motor on
`['repair','repairs']` — the SAME failure shape as the pet-insurance case: a generic
service word legitimately used by a different type's answer, mined as "distinctive" for
one type only because that's where the corpus happened to concentrate it).

**What a real retry needs, not attempted here:** mine at open-vocab-candidate
granularity (query `classify_candidate_type()` / `candidate_vocab.py`'s active-vocab
set, not just the 12 hardcoded `policy_type` values) so `pet_insurance` gets its OWN
distinctive-term bucket instead of donating to `health`'s. Until that's built, do not
re-attempt the coarse-grained version — it will reproduce this exact failure on every
open-vocab type (jewellery, workmens compensation, etc. — see
[[project_open_vocab_promotion_wired]] for the full candidate list), not just pets.

**A measurement-methodology note worth keeping, independent of the verdict above:**
correlating `[vocab_probe]` log lines back to specific corpus cases by request ORDER
alone was unreliable this run — the capture window had 29 stray (non-corpus) requests
interleaved (from other traffic hitting the same container during the run) and 18
corpus requests that never produced a `TIMING` line at all (likely a refusal/early-exit
path that doesn't reach it). What actually worked: since only 4 probe events fired
total, each was identified unambiguously by reading the `TIMING` line immediately
following it in the same request's log block (which carries the exact query text) and
looking that query up directly in `contamination_corpus.json`. This doesn't scale much
past a handful of events — a future D2 attempt with a probe that fires more often will
need either session-id tagging on the `[vocab_probe]` log line itself, or a
query-text-membership filter (not order-based) before any bulk correlation.
