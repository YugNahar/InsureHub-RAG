# Plan: fix shallow/generic answers (context-budget starvation)

## The symptom

"Can you explain travel insurance in detail" returns 6-8 numbered points of
which 4+ are content-free filler:

> 2. Coverage is usually limited, so it's important to understand the terms
>    and conditions of your policy.
> 3. Premiums can vary, and you should review them regularly.
> 4. Definitions of covered items are detailed in the terms and conditions.
> 6. The insurance contract has a defined duration, and changes can occur.

Nothing is *false* and nothing is contaminated — the answer is simply hollow.
Meanwhile the KB holds the specific material those points gesture at
(`travelinsuranceguide.pdf` p9-10 gives exact cancellation/interruption rules,
a €5,000 per-journey cap, €80/day for max 45 days; p22 lists the actual expiry
conditions). None of it reaches the model.

This is a **different failure class** from
`plan_claim_answer_correctness.md` (wrong-type imports and factual
inversions). Do not conflate them — this one is starvation, not pollution.

## Root cause: the LLM is shown 900 characters of context. Always.

Measured live 2026-08-04 (`[Compressor] budget trim: 20358 chars across 8
chunks → target 900 chars`) and reproduced arithmetically:

```
_MAX_INPUT_TOKENS      = 3900                      # multi_source_rag.py ~6388
_CHARS_PER_TOKEN       = 3                         # ~4334
_context_token_budget  = max(300, 3900 - prompt - history - output_reserve)
_context_budget        = min(6000, _context_token_budget * 3)      # ~6428
```

| Mode | Prompt template | est. tokens | output_reserve | context budget |
|---|---|---|---|---|
| DETAILED | 10,942 chars | 3,647 | 1,500 | **900** (floored) |
| STRICT | 13,312 chars | 4,437 | 300 | **900** (floored) |
| CONVERSATIONAL | 10,461 chars | 3,487 | 300 | **900** (floored) |

Every mode goes **negative** before the floor catches it (detailed:
`3900 − 3647 − 1500 = −1247`). The `max(300, …)` floor then silently rescues
the arithmetic, and 300 × 3 = the 900 chars seen in the log. The floor is not
a safety margin here — it is the operating point, on every single request.

### What 900 chars does inside the compressor

`context_compressor.py::compress_to_budget` splits fairly first:
`fair_share = 900 // 8 = 112` chars per chunk. Redistribution then has only
~4 leftover chars to hand out, so it is a no-op. Each chunk takes the
sentence-ranking path, and there:

```python
if len(top_sentence) + 2 > alloc:          # 112 chars — nearly every real sentence
    truncated = top_sentence[:alloc].rsplit(' ', 1)[0] + '…'
    → kind = "hard_truncate"
```

So the model receives **8 mid-sentence fragments of ~112 chars each**, not 8
chunks. Generic filler is the only thing that can be written from that. The
compressor is behaving exactly as designed — it is being handed an impossible
budget.

### Why this went unnoticed for so long

The symptom *was* observed and partly worked around, one layer downstream.
`multi_source_rag.py` ~6489 already carries this comment:

> compress_to_budget can shrink a multi-hundred-word chunk down to a single
> ~100-char sentence when many chunks compete for a shared budget (confirmed
> live: 8 detailed-mode chunks compressed to 64-114 chars each…)

That observation was used to justify keeping `_full_context_uncompressed` for
the **grounding checks** (so correct answers stop being falsely rejected) —
a good fix for the symptom it addressed. But the upstream question, *why is
the budget that small*, was never asked. Consequence: grounding validates
against the full text while the model only ever sees slivers.

**Strong hypothesis worth testing first:** several past "padding", "hollow
answer", "forced filler", and `_MIN_ANSWER_SENTENCES`-style bugs may be
downstream symptoms of this same starvation. If so, some existing patches
become redundant — but do **not** remove any of them speculatively; re-run
their regression cases after Phase C and remove only what provably no longer
fires.

---

## Phase A — RESOLVED (2026-08-04)

**`curl http://123.253.124.14:7000/v1/models` → `"max_model_len": 4096`.**

`_MAX_INPUT_TOKENS = 3900` is a correct, deliberate guard against the
model's real hard ceiling — not an arbitrary conservative guess. It leaves
only a 196-token safety margin below the actual limit vLLM will 400 on.

**Consequence: Phase C1 (raise `_MAX_INPUT_TOKENS`) is CLOSED.** There is no
headroom to reclaim from the model side — Qwen2.5-7B was launched here with
`--max-model-len 4096`, not the 32K the model architecture supports, and
that's outside this codebase's control (a vLLM launch-flag change, a
separate, much larger conversation about GPU memory / KV-cache sizing on
whatever's serving `123.253.124.14:7000`, not something to touch as part of
this plan).

### Phase A follow-up: the budget is ALSO being overcharged ~1,200 tokens/request

Measured 2026-08-04 against the real tokenizer (vLLM exposes `POST /tokenize`
at `123.253.124.14:7000/tokenize` — note: NOT under `/v1/`):

| Prompt | chars | `chars//3` est. | **real tokens** | phantom cost | real ratio |
|---|---|---|---|---|---|
| DETAILED | 11,207 | 3,735 | **2,523** | **+1,212** | 4.44 |
| STRICT | 13,577 | 4,525 | **3,116** | **+1,409** | 4.36 |
| CONVERSATIONAL | 10,724 | 3,574 | **2,433** | **+1,141** | 4.41 |

`_CHARS_PER_TOKEN = 3` overcharges every prompt by ~45%. Recomputing the
budget with real counts (history=0):

| Mode | today | with real token counts |
|---|---|---|
| CONVERSATIONAL | 300 tok / 900 ch (floored) | **1,167 tok / 3,501 ch — 3.9x** |
| STRICT | 300 tok / 900 ch (floored) | **484 tok / 1,452 ch — 1.6x** |
| DETAILED | 300 tok / 900 ch (floored) | −123 → **still floored** |

**Why the constant is 3, and why that is not simply wrong.** The comment at
`multi_source_rag.py` ~4420 documents a real production overflow at
`3585 input + 512 output = 4097` when the ratio was 4. That is genuine — but
the constant is doing **two jobs with opposite safety directions**:

1. **Prompt cost** (`len(PROMPT) // ratio`): a LOW ratio over-estimates cost →
   conservative. Applied to *static, known* text.
2. **Budget→chars** (`budget_tokens * ratio`): a LOW ratio packs in fewer
   chars → conservative. Applied to *variable, untrusted* KB text whose real
   ratio genuinely varies (jargon, PDF artifacts, citations, numbers).

One constant tuned safe for both compounds the conservatism. The overflow the
comment describes came from job 2 (context text), but the fix was applied to
both — and job 1 never needed a heuristic at all, because the prompt is static
text we can simply measure.

### Phase C2, step 2 — INVESTIGATED, NO SAFE CANDIDATES FOUND (2026-08-04)

Before cutting anything, checked whether the citation-leak and third-party-
brand prompt rules are actually redundant with their code guards, as C2's
original plan assumed. They are not — both guards are deliberately **narrow,
closed-list safety nets**, not substitutes for the prompt's general
instruction:

- `_CITATION_LEAK_RE` (`multi_source_rag.py` ~9981) only matches the literal
  pattern `filename.pdf (Page N)`. The prompt rule's own example — "the guide
  mentions...", a **paraphrased** self-reference with no filename/page in it
  at all — is NOT covered by this regex. Removing the prompt rule would
  delete the only defense against that leak shape.
- `_THIRD_PARTY_BRAND_RE` (~10046) is a closed list of 5 already-observed
  brand strings ("My Pages", "If Mobile", etc.). It cannot catch a brand name
  from a document not yet in the KB. The prompt rule is what prevents the
  NEXT unseen case; the code guard only mops up ones already seen once.

This is by design, not oversight — it matches the "belt-and-braces, never the
primary fix" discipline `plan_claim_answer_correctness.md` Phase 5 states
explicitly for this exact rule pattern. **Conclusion: C2 step 2 (remove
code-covered rules) has no safe candidates in this codebase.** Every code
guard checked is a reactive safety net for the general, proactive prompt
instruction, not a replacement for it. Do not remove either rule.

Given C0+C3 already delivered the load-bearing wins (3.9x conversational,
2.6-2.9x detailed budget) and C2 is confirmed no longer load-bearing, the
remaining C2 work (step 1: move pure historical backstory to code comments
while KEEPING the concrete example each rule uses for in-context calibration;
step 3: light wording compression without losing substance) is real but
lower-value and higher-risk-per-token-saved than C0/C3 were. Deferred pending
explicit go-ahead rather than pushed through without one, given the
non-trivial risk of quietly degrading a small model's output quality for a
now-marginal token saving.

---

### Phase C3 — DEPLOYED AND VERIFIED (2026-08-04)

**Finding before implementing:** `_output_reserve = 1500 if detailed else 300`
was never tied to what the completion request actually asks for. The live
`.env` has `VLLM_MAX_TOKENS=512` (not even the 900 code-default) — so detailed
mode was reserving **1500 tokens against a request that only ever asks for
512**, wasting ~988 tokens (~25% of the whole 3900-token budget) on a phantom
reservation.

**Fix:** `_VLLM_MAX_TOKENS_DETAILED`/`_VLLM_MAX_TOKENS_BRIEF` now computed
ONCE at import (`os.getenv`, same defaults as before) and used for BOTH the
completion request's own `max_tokens` payload field AND `_output_reserve` —
one shared constant, structurally cannot drift apart again. Also replaced the
`__import__("os").getenv(...)` idiom at the completion-call site with the
shared constant (dead-code cleanup, `os` was already imported at module
level).

**Recomputed detailed-mode budget:** `3900 - 2523(real prompt tokens, Phase
C0) - 0(history) - 512(now-correct reserve) = 865` → **2595 chars**, no
longer floored (vs. 900 before either fix).

**Truncation-risk check (the real danger of cutting `_output_reserve` — see
plan intro):** added a permanent `finish_reason` log line for detailed mode
(previously only brief mode had this signal). Ran 3 fresh detailed queries
post-deploy (travel insurance, motor claims, health insurance) — all three
returned `finish_reason='stop'`, none `'length'`. Measured the real token
count of the longest completion (9-point health-insurance answer, 2378
chars) via `/tokenize`: **356 / 512 tokens — 70% of ceiling, ~30% margin**.
This is empirical evidence of safety, not a full p95 study (3 samples, one
session) — the new finish_reason log line stays in place permanently as an
ongoing signal. **If `'length'` starts appearing in production logs, that's
the trigger to revisit `VLLM_MAX_TOKENS` itself** — a harder, separate
problem, since raising it directly costs latency on a ~7-8 tok/s backend.

---

### CRITICAL INCIDENT — found and fixed live during Phase D verification (2026-08-04)

**What happened:** while spot-checking two of the claims-plan's regression
cases with fresh phrasings post-Phase-D, both returned "Could not generate
an answer due to an internal error." Real vLLM 400s in the logs:

> This model's maximum context length is 4096 tokens. However, you requested
> 300 output tokens and your prompt contains at least 3797 input tokens, for
> a total of at least 4097 tokens.

This is the exact production crash the entire dynamic-budget mechanism
exists to prevent — reproduced live, caused by this session's own changes.

**Root cause:** `_context_budget` (the char figure `compress_to_budget` is
given) is computed from a char-based estimate of what goes into
`context={full_context}` — but `full_context` as actually assembled
(`multi_source_rag.py` ~6746) is NOT simply the compressed chunks joined
together. It also includes, per chunk, a `"[Document: file (Page N)]\n"`
label; `_rerank_windows()` re-windowing the chunk's text against the query
(which can return a different span than what `compress_to_budget` trimmed
to); and conditionally a `"[Related prior answers]"` block. None of that
overhead was in the char estimate. **This gap already existed before today**
— it wasn't introduced by C0/C3/D. It was silently absorbed by the phantom
slack Phase C0 and C3 removed (the ~1,200-token prompt overcharge, the
~988-token output-reserve overcharge). Fixing the real waste correctly
also deleted the safety margin that had been accidentally hiding this
separate, pre-existing bug. Confirmed on a genuinely fresh, single-turn,
brief-mode query — not a long-history edge case.

A second, independent, pre-existing inconsistency compounded it: the
non-streaming fallback path (`get_insurance_llm(temperature=0)`, called with
no `max_tokens` argument) used `router.py`'s own default
(`VLLM_MAX_TOKENS` env var = 512) regardless of whether the original request
was brief or detailed — never the 300 the streaming attempt had actually
asked for. Both paths were hitting the same wall from slightly different
numbers.

**Fix — a real-token safety valve, not a better estimate:** rather than
trying to account for every source of char-vs-token slop (fragile, will
drift the next time any of those pieces changes), measure the REAL token
count of the fully-assembled `prompt` string via `_measure_prompt_tokens()`
(the same `/tokenize` call Phase C0 built) immediately before sending, and
dynamically cap `max_tokens` so `real_prompt_tokens + max_tokens` structurally
cannot exceed 4096 (50-token buffer — tight, because this is a real
measurement, not a heuristic). Applied to BOTH the streaming payload and the
fallback path (`get_insurance_llm(temperature=0, max_tokens=_safe_max_tokens)`
— also fixes the second inconsistency above as a side effect, since both
paths now share one authoritative value). Logs a warning whenever the safety
valve actually reduces the output budget, so this is visible going forward
rather than silently eating into answer length.

**Verified:** both originally-crashing queries now complete successfully
(logs show `real prompt tokens=3973`/`3783`, correctly capped to
`73`/`263` of the intended `300` budget — no crash, shorter but complete
answers). Re-ran the claims-plan's other fix (motor stolen-vehicle-transfer)
in the same batch — still correctly dropped by Phase 1, confirming the fix
didn't disturb that mechanism.

**Follow-up survey (6 varied queries, not exhaustive):** the gap fired on
1 of 6 fresh queries, 63/512 tokens (minor). Not universal or catastrophic,
but real and worth root-causing properly at some point — likely candidates
are the per-chunk `[Document: ...]` label overhead (small, additive per
chunk) and/or `_rerank_windows()` returning more than what was trimmed to
(potentially the larger factor). **Not fixed here** — the safety valve makes
it safe (never crashes) but not optimal (occasionally shorter completions
than intended when the gap is large). Flagged as a candidate for a future,
separate investigation — out of scope for this incident fix, which was about
stopping the crash, not perfecting the estimate.

---

### Direct before/after against the ORIGINAL reported symptom (2026-08-04)

Fresh generation post-C0+C3 (`target 2595 chars`, confirmed via
`Compressor` log — exact match to the predicted 865×3), "Please describe
everything about travel insurance policies":

> 5. Travel insurance coverage is usually limited to the period of your
>    travel.
> 6. The insurance covers compensation for journey cancellations due to
>    certain events, such as natural disasters or terrorist attacks.
> 7. Loss or damage to luggage is covered under the insurance policy.
> 8. The insurance does not cover illnesses or injuries that occurred or
>    began before the start of the journey.

vs. the ORIGINAL symptom this plan opened with:

> 2. Coverage is usually limited, so it's important to understand the terms
>    and conditions of your policy.
> 3. Premiums can vary, and you should review them regularly.
> 4. Definitions of covered items are detailed in the terms and conditions.

Points 3-4 in the new answer are still somewhat generic (definitional
boilerplate — "a legal contract," "premium each term"), but points 5-8 are
now genuinely specific, grounded facts that were structurally impossible to
surface at a 900-char budget: the travel-period limit, named cancellation
triggers, luggage coverage, and a real pre-existing-condition exclusion
clause. This is not a reworded version of the same hollow answer — it
contains real facts (luggage coverage, the exclusion clause) that were never
in ANY prior generation of this query this session. Direct evidence the fix
addresses the actual reported symptom, not just the compressor's internal
metrics.

**Testing-discipline note confirmed live:** the first re-test attempt hit a
stale KV-cache entry from an earlier near-identical phrasing tested during
C0 verification (before C3 deployed) — zero `Compressor` log line appeared
at all, because a cache hit short-circuits before retrieval runs. Exactly
the trap documented in the claims plan's testing-discipline section. Confirms
that discipline generalizes across both plans, not just the one it was
written for.

---

### Phase C0 — DEPLOYED AND VERIFIED (2026-08-04)

Implemented in `multi_source_rag.py`: `_measure_prompt_tokens()` calls
`POST {VLLM_HOST}/tokenize` once per prompt at module import time (imports
`VLLM_HOST`/`VLLM_API_KEY`/`_resolve_vllm_model` from `router.py`, no
circular import), falls back to `len(p)//_CHARS_PER_TOKEN` on any exception.
`_context_budget`'s own ratio-3 conversion left untouched, exactly as
specified.

**Boot log:** `real prompt token counts — strict=3116 detailed=2523
conversational=2433 (vs. char/3 estimates: 4525/3735/3574)` — matches the
manual measurement exactly, fallback never triggered.

**Live verification, conversational mode** ("What is a no claim bonus?"):
`[Compressor] budget trim: 15138 chars across 5 chunks → target 3501 chars`
— exact match to the predicted 3.9x (900→3501).

**Live verification, detailed mode** ("Can you explain travel insurance in
full detail?"): still `target 900 chars` — also exactly as predicted (
`2523 + 1500(output_reserve) - 3900 = -123`, still floored). This is not a
bug; it's the documented reason Phase C3 (`_output_reserve`) is needed next
for detailed mode specifically — C0 alone was never going to fix it.

No errors or fallback-path warnings in logs after deploy. `contamination_corpus_runner.py` grounding checks are unaffected (they already
read `_full_context_uncompressed`, independent of the compressed budget).

---

### NEW Phase C0 — decouple the ratio (do this BEFORE C2)

Replace the prompt-cost estimate with a **real tokenizer count**, keep the
conservative ratio for the context conversion:

- `_STRICT/_DETAILED/_CONVERSATIONAL_PROMPT_TOKENS_EST` ← measured token count.
- `_context_budget = _context_token_budget * _CHARS_PER_TOKEN` ← **unchanged
  at 3.** Do not touch this; it is the one guarding against the documented
  overflow.

**Implementation constraint (verified, do not skip):** there is no locally
cached Qwen tokenizer — `AutoTokenizer.from_pretrained` inside the container
fails with `OSError: couldn't connect to huggingface.co`, and `HF_HUB_OFFLINE=1`
is set deliberately so boot never depends on HF reachability
(`project_hf_offline_mode_container_boot`). So:

- Compute lazily/at import via one `POST /tokenize` call to the vLLM host —
  a server this app already hard-depends on for every single request, so it
  adds no new failure mode — and
- **fall back to the current `len(p) // _CHARS_PER_TOKEN` on any exception.**
  Worst case must be exactly today's behaviour, never worse. Cache in a module
  global; never re-measure per request.
- Self-maintaining by construction: if a prompt is edited, the next boot
  re-measures. This is the failure mode the old hardcoded 700-token constant
  had (it "drifted badly out of date and caused a real production crash" —
  same comment block) so do NOT hardcode the measured numbers.

### Revised priority

1. **Phase C0 (new) — decouple the ratio.** Cheapest, lowest-risk, largest
   single win: ~3.9x context for conversational (the most-travelled path),
   ~1.6x for strict, no prompt rewriting, no new failure mode.
2. **Phase C3 — right-size `_output_reserve`.** Promoted from last to second,
   because in DETAILED mode it is the *dominant* term: even with exact token
   counts, `2,523 + 1,500 > 3,900` keeps detailed mode floored. Detailed
   answers measured this session run ~800-900 chars ≈ 200 tokens, so 1,500 is
   ~7x over-provisioned; cutting to ~600 would free ~900 tokens and put
   detailed mode at ~2,331 chars (**2.6x**). Still measure p95 completion
   tokens before changing it — under-reserving truncates real answers.
3. **Phase C2 — shrink the prompt templates.** Still worth doing (the embedded
   "Confirmed live: …" debugging narratives genuinely should not be shipped
   per-request), but it is no longer load-bearing and no longer the only
   lever — so it can be done carefully and incrementally rather than under
   pressure.
4. **Phase D — compressor degradation strategy.** Unchanged in value.
5. **Phase B — log the floor loudly.** Ship anytime; one line, zero risk.

**Correction to the earlier Phase A note:** its claim that "the prompt's own
size is the only significant lever left" was wrong — it was written before the
tokenizer was measured. Two larger levers exist (C0 and C3), both cheaper than
C2. Superseded text kept below for history.

---

~~This makes **Phase C2 (shrink the prompt templates) the primary, load-
bearing fix, not one option among several** — with 4096 total tokens and a
fixed model-side ceiling, the prompt's own size is the only significant lever
left before context is affected.~~ *(superseded — see Phase C0 above)*
Phase C3 (right-size `_output_reserve`) and
Phase D (compressor degrades by dropping tail chunks, not fragmenting all of
them) both matter more too, for the same reason. Re-prioritized:

1. **Phase C2 first** — every token trimmed from the ~10.5-13.3KB prompts is
   a token returned directly to context, with zero downside once verified
   (the debugging narratives and code-guard-duplicated rules were never
   supposed to cost tokens on every request in the first place).
2. **Phase D second** — even after C2, 900+ chars is still a tight budget
   for a detailed answer citing several chunks; make what's left work
   better rather than fragmenting everything equally.
3. **Phase C3 last, and only after measuring** — `_output_reserve` is the
   one lever that can make things WORSE if cut wrong (truncated answers),
   so it needs real completion-token data before touching, not estimation.
4. **Phase B (log the floor loudly)** — ship immediately regardless of the
   above three, it's a one-line, zero-risk addition.

---

## Phase A — Find the real ceiling (BLOCKING; do nothing else first)

`_MAX_INPUT_TOKENS = 3900` is suspiciously ~4096. If the remote vLLM
(`123.253.124.14:7000`, `Qwen/Qwen2.5-7B-Instruct-AWQ`) was launched with
`--max-model-len 4096`, then 3900 is a correct guard and raising it causes
hard 400s — the exact production crash `project_prompt_token_budget_overflow`
exists to prevent. But Qwen2.5-7B supports 32K natively, so this may be 8x
headroom left on the table for no reason.

```bash
curl -s http://123.253.124.14:7000/v1/models | python3 -m json.tool
```

vLLM reports `max_model_len` there. **Every later phase's sizing depends on
this number.** Record it in this file.

- If `max_model_len` ≫ 4096 → Phase C1 (raise the budget) is the primary fix
  and is nearly free.
- If `max_model_len == 4096` → C1 is closed; C2/C3/D carry the whole load.

Do not assume. Do not raise `_MAX_INPUT_TOKENS` before this returns.

---

## Phase B — Make the floor loud (cheap, do immediately, independent of A)

**File:** `multi_source_rag.py` ~6424-6428.

The `max(300, …)` clamp currently hides a pathological state. Add a
`logger.warning` when the pre-clamp value is at or below the floor, naming
the culprit sizes (prompt tokens, history tokens, output reserve, the
resulting negative number). One line, zero behaviour change, and it means no
future prompt growth can silently starve context again.

Ship this even if everything else is deferred.

---

## Phase C — Reclaim budget (order matters; each is independently shippable)

### C1. Raise `_MAX_INPUT_TOKENS` — only if Phase A permits
Set it from the measured `max_model_len` with a real safety margin (e.g.
`max_model_len − 1000`), not a hardcoded guess. Keep the floor and Phase B's
warning as the backstop. Highest impact, lowest effort — **if** A allows it.

### C2. Shrink the prompt templates
**File:** `prompt_template.py`. 10.5-13.3 KB per prompt is enormous, and it is
paid on **every request, forever**. Three safe reductions, in order:

1. **Strip the embedded debugging notes.** Rules accumulated over months carry
   "Confirmed live: <a specific past bad answer>" narratives *inside the
   prompt string*. Those belong in a code comment, not in tokens shipped to
   the model on every call. This alone is likely several hundred tokens.
   Move each to a `#` comment directly above the string.
2. **Drop prompt rules that a deterministic code guard already enforces.**
   Citation-leak, third-party-brand, and the always-false-claim families are
   all enforced post-generation in `ask_stream` and will be stripped whether
   or not the model complies. `project_dont_trust_buried_disclosure_instructions`
   establishes this small model ignores such instructions under pressure
   anyway — so the prompt copy is paying full token price for near-zero
   enforcement value. **Remove the prompt copy only where a code guard
   demonstrably covers it**, and re-run that guard's regression case after.
3. **Compress surviving rule wording.** Terse imperatives, no worked examples.

Target: get each template under ~5,000 chars (~1,650 tokens). That alone
returns ~1,900 tokens ≈ 5,700 chars of context in detailed mode.

### C3. Right-size `_output_reserve`
1500 tokens (detailed) is provisioned for an answer that measures ~800-900
chars ≈ 300 tokens in practice. Before changing it, **measure**: pull actual
completion-token counts for detailed answers from the vLLM responses/logs
across a spread of queries, take the p95, add margin. Do not eyeball this —
under-reserving truncates real answers, which is a worse bug than the one
being fixed.

---

## Phase D — Make the compressor degrade sensibly, not uniformly

**File:** `context_compressor.py::compress_to_budget`.

Independent of how much budget Phases A/C win back, the fair-share strategy
has a genuine design flaw at the low end: it spends the budget making *every*
chunk unusable rather than keeping *some* chunks usable.

**Fix:** establish a minimum viable per-chunk allocation (a full sentence —
suggest ~200-250 chars, calibrate against `_MIN_SENT_CHARS = 25` and observed
KB sentence lengths). When `max_total_chars // n` falls below it, **reduce
`n`**: drop the lowest-ranked chunks entirely and give the survivors a real
allocation, instead of fragmenting all of them.

Rationale: 3 coherent chunks beat 8 mid-sentence fragments for both answer
quality and grounding. Chunks arrive pre-sorted by relevance, so "drop the
tail" is already well-defined here.

Two guards, both from this repo's own history:
- The existing fair-share behaviour was itself a deliberate fix for "10 chunks
  competing for 6000 chars left chunks 5-10 with nothing" (see the method's
  docstring). **Do not simply revert to rank-order fill.** The new rule must
  only engage when fair-share is below the viability floor — above it, current
  behaviour must be preserved byte-for-byte.
- Dropping chunks changes what `_full_context_uncompressed` should contain.
  Confirm whether grounding should still see dropped chunks (probably yes —
  it reads the uncompressed text and is unaffected) and state the answer in a
  comment, so the next reader doesn't have to re-derive it.

---

## Phase E — Verification (do not skip; two of these are cross-plan)

1. **Re-run the exact travel query.** `corrected_text` should contain the
   specific figures/conditions the KB has (cancellation/interruption rules,
   the €5,000 cap, real expiry conditions) instead of "premiums can vary".
   Numbers-in-answers policy note: `feedback_numbers_only_in_examples` means
   currency figures are deliberately suppressed unless the query asks for an
   example — so judge on **specificity** (named conditions, real exclusions,
   concrete steps), not on whether € amounts appear.
2. **Latency check.** More context = more prompt tokens = slower TTFT on a
   ~7-8 tok/s backend. Compare against the existing latency baseline harness.
   A large quality win may justify some regression; an unmeasured one does not.
3. **Re-run the contamination corpus** (`contamination_corpus_runner.py
   --repeats 5`). More context per chunk plausibly *raises* cross-type
   contamination — the two plans interact directly here. The claims sub-corpus
   from `plan_claim_answer_correctness.md` Phase 0 must be re-baselined after
   this lands, or its numbers become incomparable.
4. **Re-check the "hollow answer"/padding regression cases** referenced in the
   hypothesis above, to see which existing patches are now redundant. Remove
   only what provably no longer fires.

## Testing discipline

Same as `plan_claim_answer_correctness.md` — in particular: judge from the
trailing `corrected_text`, never the raw stream; use a fresh query phrasing
and confirm `KV cache stored` (not `hit`) in the logs; deploy with
`docker compose up -d --force-recreate --no-deps api`. Note
`DISABLE_QUERY_CACHE` is currently **1** in `.env` for the corpus sweep and
must be restored to `0` afterwards.

## Critical files

- `RAG_InsureAI/app/multi_source_rag.py` — `_MAX_INPUT_TOKENS` ~6388,
  `_CHARS_PER_TOKEN`/`*_PROMPT_TOKENS_EST` ~4326-4337, budget computation
  ~6424-6428, `compress_to_budget` call site + `_full_context_uncompressed`
  ~6483-6509
- `RAG_InsureAI/app/context_compressor.py` — `compress_to_budget` (fair-share
  ~186-198, hard-truncate path ~337-342)
- `RAG_InsureAI/app/prompt_template.py` — the three templates
- `RAG_InsureAI/contamination_corpus_runner.py` — cross-plan regression gate
