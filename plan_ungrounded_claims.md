# Plan — ungrounded-claim leakage (the class `plan.md`'s contamination work does not cover)

**Author:** Opus 5 analysis pass, 2026-07-28. Written for implementation by another
engineer/model — **this document contains no implementation code by design.**

**Status:** Assessment complete, fixes specified, not yet implemented.

---

## 1. Verdict on the existing cross-topic contamination fixes

**They work for the class they were built for, and that class does not include the
error the user just reported.** This is not a regression — it is a scope boundary that
was documented at the time and has now been hit in production.

What is currently active (per `plan.md` and confirmed by reading the code):

| Layer | Where | Status |
|---|---|---|
| Phase 1 window-scoped generation context | `multi_source_rag.py` | active |
| `_TYPE_GIVEAWAY_TERMS` jargon filter | `multi_source_rag.py:1298-1310`, applied in `_text_has_giveaway_contamination` `:1330` | active |
| `_EXCLUSION_LANGUAGE_RE` exemption | `multi_source_rag.py:1323` | active |
| Ungrounded-**point** filter (`_point_grounded`) | `multi_source_rag.py:7047-7056` | active |
| Ungrounded/mis-scoped **currency** filter | `multi_source_rag.py:~7879-8050` | active |
| Phase 2 semantic per-point relevance gate | — | **built, 280-run test, zero benefit, reverted off** |
| Phase 3 retire hardcoded filters | — | **skipped** (its precondition was Phase 2 proving out) |

So the live defense against cross-topic bleed is a **closed hardcoded phrase list plus a
word-overlap grounding ratio**. Both are defeated by the reported error, for reasons that
are mechanical and provable, not speculative.

---

## 2. The reported error, audited claim by claim

Query: *"Can you explain motor insurance in detail?"*
Retrieved sources: `9.3 INSURANCE LAW AND PRACTICE.pdf` p194/p201, `m4-3f.pdf` p8/p13/p15/p16.

| # | Claim in the answer | Verdict | Class |
|---|---|---|---|
| 1 | contract between insured and insurer covering vehicle risks | OK | — |
| 2 | insurable interest (benefit from safety / suffer loss) | OK | — |
| 3 | covers repairs, injured third party, "payment for a rented vehicle" | **unverified** — could not confirm rental/courtesy-car cover in retrieved text | possible fabrication, needs a targeted check |
| 4 | "vehicles are categorized based on their market value at the time of loss" | **garbled** — KB categorises by cubic capacity, IDV, zone, age (`m4-3f.pdf` p9) | concept swap |
| 5 | **"The premium is calculated based on the market value of the vehicle at the time of loss"** | **WRONG, directly contradicted** | **within-topic factual inversion** |
| 6 | **"claim form along with original bills and discharge summaries"** | **FABRICATED** | **lexical-anchor hallucination** |
| 7 | **"condition of average … insured shares the cost of repairs according to the extent of their liability"** | **MISDEFINED + mis-scoped** | **definition corruption** |
| 8 | "designed to protect the insured against various risks associated with their vehicle" | tautology | padding |

### Evidence for #5
`m4-3f.pdf` p9 states rating is on *"cubic capacity … Insured's Declared Value (IDV), the
Zone of operation and age of the vehicle."* Market value at time of loss is the
**claim-settlement** basis (`9.3 INSURANCE LAW…` p229, `m4-3f.pdf` p13), not the premium
basis. The answer took a real sentence and attached it to the wrong side of the
policy lifecycle.

### Evidence for #6 — the important one
- `"discharge summar"` appears in **0 chunks in the entire knowledge base** (verified
  across all four `turbovec_data/**/*.ndjson` stores).
- The retrieved motor chunk `m4-3f.pdf` p13 contains **`"Discharge voucher (full and
  final payment)"`** and **`"Receipted bill from the repairer"`**.
- So the model took the genuine token `discharge` out of *discharge voucher* (a
  claim-payment acknowledgement) and re-formed it into *discharge summaries* (a hospital
  document), pairing it with *original bills* — phrasing that matches the **health**
  claim checklist in `m4-5f.pdf` p7/p8 (*"Hospital receipts/ original bills … Hospital
  admission and discharge slip"*), a document that **was not retrieved for this query**.

This is neither classic cross-topic contamination (no wrong-topic chunk was retrieved)
nor free-floating hallucination (the anchor word *was* in context). It is a third thing:
**a grounded word re-assembled into an ungrounded, wrong-domain phrase.**

### Evidence for #7
KB (`9.3 INSURANCE LAW…` p200, p212): condition of average / average clause is the
**underinsurance** doctrine — if the sum insured is less than actual value the insured
"shall be considered as being his own insurer for the difference, and shall bear a
rateable proportion" — and is scoped *"primarily in property claims – fire and
engineering."* The answer redefined it as sharing repair costs "according to the extent
of their liability," which is a different concept, and applied it to motor unqualified.

---

## 3. Why each existing defense misses this — mechanically

### 3.1 `_TYPE_GIVEAWAY_TERMS` cannot fix #6, and adding the phrase to it is a provable non-fix

Two independent reasons:

1. `"discharge summary"` is not in the `health` tuple (`:1300` has only *health
   insurance, domiciliary hospitali[sz]ation, cashless hospitali[sz]ation, pre-existing
   disease*).
2. **More importantly** — `_text_has_giveaway_contamination` at `:1374` does
   `if not any(term in context_lower for term in _terms): continue`. The filter
   **requires the giveaway term to be present in the retrieved context** before it will
   treat a match in the answer as contamination. `"discharge summary"` is in **no** chunk
   anywhere, so the term can never be found in context, so the check will `continue` past
   it **even if the phrase is added to the list.**

> **Do not "fix" this by appending to `_TYPE_GIVEAWAY_TERMS`.** It is structurally
> incapable of firing here. That design choice is deliberate and correct for its own
> purpose (it prevents false positives on legitimate mentions) — it simply makes the
> mechanism blind to *invented* vocabulary by construction.

### 3.2 `_point_grounded` is defeated by overlap dilution — and its threshold is weaker than its comment claims

`_matched_words` (`:7039`) counts **individual content words (len ≥ 4) present in the
context word set**. The point in question —

> *"The claim procedure involves submitting a claim form along with original bills and
> discharge summaries."*

— has roughly 11 distinct content words, of which *claim, procedure, involves,
submitting, form, with, original, bills, discharge* all genuinely appear in the retrieved
motor text. Only **`summaries`** is absent. One ungrounded word out of eleven cannot move
a ratio.

Worse, the effective bar is lower than intended. The threshold at `:7052-7053` is
`min(_MIN_MATCHES, max(1, ceil(total/2)))` with `_MIN_MATCHES = 4`. The `min()` means the
50% fraction only binds for points with **≤ 8 content words**; for anything longer the bar
is a **flat 4 matched words**. A 20-content-word point passes at 20% grounding. The
comment block above it (`:7028-7035`) describes the rule as a 50% floor, which is not what
the code does for long points.

**This discrepancy should be resolved deliberately** — either the comment is wrong, or the
`min()` should be a `max()`. Do not change it silently; see §5.3, it needs its own
measurement because tightening it will drop currently-kept content.

### 3.3 The currency filter already solved the right *shape* of problem — for numbers only

`_currency_grounded` extracts a token class (`_CURRENCY_RE`), requires its digits to be
present in context, and drops the unit otherwise. `_qualifier_mismatched` then adds
scenario-scoping on top. **This is exactly the correct architecture** for #6, restricted
to numbers.

`project_currency_qualifier_mismatch.md` already recorded this exact gap as known and
deliberately deferred:

> *"a mismatched claim mentioned WITHOUT an attached currency figure … isn't caught by
> this mechanism at all, since the whole check is keyed off `_CURRENCY_RE` matches."*

This plan closes that documented gap.

### 3.4 The regression corpus cannot currently catch it

`contamination_corpus.json` grades by **closed `forbidden_phrases` lists** per case. The
`motor-health-domiciliary-01` case lists *domiciliary/cashless/pre-existing* — not
*discharge summaries*. A corpus keyed to previously-observed vocabulary is structurally
unable to detect newly-invented vocabulary. This is the same fragile-fixed-list pattern
recorded in `project_fragile_signal_lists.md`.

---

## 4. Proposed fixes

Three separate items, deliberately ranked. **Fix A is the recommended one to do first**
— it is deterministic, high-precision, and closes a documented gap. Fixes B and C are
lower-confidence and should not be bundled with A.

### Fix A — Artifact-noun grounding filter (targets #6) — **recommended**

**Idea.** Generalise the *shape* of the currency-grounding check from numbers to a narrow,
curated class of **named documents / instruments / artifacts**. If the answer names a
specific document, that document's name must actually appear in the retrieved context.

**Why this is safe where a general fact-checker would not be.** It never makes a semantic
judgement. It only asks a literal question — *"is this specific artifact noun present in
the retrieved text?"* — over a **closed, curated list**, and fails open on anything not in
the list. That is the same contract as `_currency_grounded`, which has been running
safely.

**Where to implement.**
- File: `RAG_InsureAI/app/multi_source_rag.py`
- Location: **inside the existing currency-filter block** (currently ~`:7879-8050`),
  as an additional predicate in the same `for _unit in _units:` loop that already
  computes `_kept_units` / `_dropped_num` (~`:8042-8049`).
- **Do not add a new independent post-processing pass.** That loop already solves the
  hard downstream problems — numbered-list renumbering, trailing-emoji punctuation, the
  lead-in re-add after dropping the first unit, and the interaction with the hollow-answer
  detector. A parallel pass would have to re-solve all of them and would be the most
  likely source of a regression.

**What to add.**
1. A module-level curated mapping of artifact nouns, near `_TYPE_GIVEAWAY_TERMS`
   (`:1298`) so the two brittle-by-nature lists live together and are reviewed together.
   Contents: multi-word names of claim/policy documents and instruments that appear in
   this KB — e.g. discharge voucher, discharge summary, discharge slip, claim form,
   cover note, surveyor's report, receipted bill, cash memo, policy schedule, bill of
   lading, proposal form, discharge certificate, death certificate, post-mortem report,
   FIR, driving licence, registration certificate. Include the *health* artifact names
   explicitly, since those are the ones most likely to be hallucinated into non-health
   answers.
2. A predicate that, for each answer unit, finds which listed artifact nouns the unit
   mentions, and returns "ungrounded" only if a mentioned artifact noun is **absent from
   `_full_context_uncompressed`**.
3. Reuse the de-wrapped context copy introduced for the currency window check
   (`_dewrapped_ctx`) — the same PDF single-`\n` line-wrap artifact will otherwise split
   two-word artifact names like `"discharge \nvoucher"` and cause false "absent" verdicts.
   **This is essential; skipping it will make the filter over-fire.**

**Matching rules (precision guardrails — all of these matter).**
- Normalise singular/plural before comparing (`summary`/`summaries`,
  `voucher`/`vouchers`, `bill`/`bills`) — otherwise *discharge voucher* in context will
  not match *discharge vouchers* in the answer and you will drop correct content.
- Normalise possessives/spacing (`surveyor's` / `surveyors` / `surveyor s`).
- Match on the **whole multi-word phrase**, never on the head noun alone. `"discharge"`
  alone must never trigger anything — that word is legitimately in the motor context.
  This is the entire point of the fix: *discharge voucher* is grounded, *discharge
  summary* is not, and only phrase-level matching can tell them apart.
- **Fail open.** Unit mentions no listed artifact → keep. Ambiguity → keep.
- Respect the existing never-drop-everything guardrail (`if _dropped_num and
  _kept_units:` at `:8016`) so an answer can never be emptied by this filter.
- Log every drop with the offending phrase, at the same level as the existing
  `"dropped %d unit(s) containing an ungrounded currency figure"` message, so drops are
  auditable in the contamination trace.

**Expected behaviour on the reported case.** The unit mentions *discharge summaries*;
normalised to *discharge summary*; absent from context (0 KB chunks); the unit is dropped;
the remaining seven points are renumbered by the existing logic.

---

### Fix B — Premium-basis vs claim-basis confusion guard (targets #5, partially #4)

**Confidence: medium. Do not bundle with Fix A.**

This is a *within-topic* factual inversion, not a grounding failure — every word in
"premium is calculated based on the market value at the time of loss" is in the retrieved
context, just wired to the wrong subject. No grounding-style filter can catch it.

Two candidate routes, in preference order:

- **B1 (recommended, cheap): a prompt rule.** `prompt_template.py` already carries
  rules of exactly this shape (see the existing rule 8(d)/13(d) about attributing
  features to the right named product). Add a rule stating that premium/rating basis and
  claim-settlement basis are distinct and must never be interchanged, with the confirmed
  live example (rating = cubic capacity/IDV/zone/age; settlement = market value at time
  of loss). Cheap, zero false-positive surface, but — per this project's own repeated
  experience — prompt rules alone are unreliable on a small model.

- **B2 (deterministic backstop, only if B1 measurably underperforms): a paired-concept
  guard.** Detect an answer clause asserting *premium … based on/calculated on … market
  value*, and check whether the retrieved context instead states the rating basis in
  different terms (IDV/cubic capacity/zone/age). Treat as an always-false claim, in the
  same spirit as `project_always_false_claim_corrections.md`. Narrow, high precision, but
  it is one more hardcoded pattern — accept the maintenance cost knowingly.

---

### Fix C — "Condition of average" definition correction (targets #7)

**Confidence: medium-low. Smallest, most self-contained item.**

The term has a single fixed correct meaning in this KB (underinsurance → rateable
proportion) and the answer stated a different one. This is the same shape as the already
fixed TPA misdefinition recorded in `project_always_false_claim_corrections.md`, so an
existing mechanism and precedent already exist — extend that, do not invent a new one.

Also consider whether the KB's own scoping (*"primarily in property claims – fire and
engineering"*) should be preserved when the term surfaces in a motor answer. Recommend
**not** hard-blocking it for motor — the doctrine can apply to motor own-damage, and a
hard block risks suppressing correct content. Correct the definition; leave the scope
alone.

---

## 5. Verification — required before any of this is called done

### 5.1 Pre-flight, before writing the filter
Build a standalone script against the **real** extracted KB text (not hand-typed
strings) covering at minimum:
- the exact failing unit (*discharge summaries*) → **must drop**;
- a unit naming *discharge voucher* → **must keep** (this is the discriminating case; if
  it drops, phrase matching or de-wrapping is wrong);
- a unit naming *claim form* / *receipted bill* / *cash memo* → **must keep**;
- a unit with a plural/possessive variant of a grounded artifact → **must keep**;
- a unit naming no artifact at all → **must keep**;
- a health-topic answer legitimately naming *discharge slip* with the health doc
  retrieved → **must keep**.

The currency fix earlier in this session needed **three** design iterations because the
first two passed a hand-written test and failed on real KB text. Test against real text
from the start.

### 5.2 Live verification
The reported failure is **nondeterministic** — it reproduced roughly 2 times in ~25 runs
of the same query earlier in this session. Consequences:
- A handful of clean runs is **not** evidence the fix works. Budget **20+ repeats**.
- **Assert on `corrected_text`, not on the streamed body.** The streamed tokens are emitted
  *before* post-processing; a filter that is working correctly still shows the bad text in
  the raw stream. An earlier check in this session reported false "no hit" results for
  exactly this reason. `frontend/public/app.js:573` confirms the UI renders
  `correctedText` when present, so `corrected_text` is what the user actually sees.

### 5.3 Regression gate
- Run `contamination_corpus_runner.py --repeats 5`. **Hard fail** on any clean-control
  contamination; warn line for repro rate is ~8.5%.
- **Extend the corpus** with a new case for this failure, otherwise it is unprotected
  against recurrence. Note that the corpus's `forbidden_phrases` design means it can only
  catch vocabulary someone has already seen — record this limitation in the corpus
  description rather than pretending the new case closes the class.
- If §3.2's `min()`/`max()` threshold discrepancy is touched **at all**, it needs its own
  before/after sweep — it will change how much content is dropped across every detailed
  answer, and it is not part of Fix A. Keep it as a separate commit.

---

## 6. Explicitly out of scope

- **Do not rebuild the Phase 2 semantic per-point relevance gate.** It was built, measured
  at 280 runs, showed no measurable benefit, and was reverted
  (`plan.md`; `project_point_relevance_ratio_to_max_unsafe.md`). Nothing in this analysis
  is new evidence in its favour.
- **Do not attempt general LLM-based fact verification of every claim.** That is the
  unbounded version of this problem, it is what Phase 2 effectively failed at, and the
  latency budget will not absorb it.
- **Do not extend `_TYPE_GIVEAWAY_TERMS` for #6** — see §3.1, it cannot fire.

---

## 7. Recommended sequencing

1. Fix A alone, with §5.1 pre-flight, §5.2 20+ repeat live check, §5.3 corpus gate. Commit.
2. Add the corpus case for this failure. Commit.
3. Fix C (small, precedented). Commit.
4. Fix B1 (prompt rule) and measure. Only consider B2 if B1 demonstrably fails.
5. Decide the §3.2 threshold discrepancy separately, on its own measurement.

Keep these as **separate commits**. Several fixes have already landed today in this area;
bundling them makes it impossible to attribute a later regression.
