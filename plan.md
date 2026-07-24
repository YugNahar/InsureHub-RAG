# Plan — Kill cross-topic contamination in point-based answers at the root

**Author:** Opus (planning only — no code here)
**Implementer:** Sonnet
**Status:** Phases 0-1 done and active. Phase 2 built, tested at real scale (280 runs), and closed — no measurable benefit, reverted to off. Phase 3 skipped (its own precondition wasn't met). Phase 4 done — `contamination_corpus_runner.py` is the standing regression command. See each phase's OUTCOME/DONE note below for details.
**Scope:** `RAG_InsureAI/app/` — primarily `multi_source_rag.py`, with touches to `prompt_template.py` and a new diagnostics module. No frontend, no ingestion re-run required for the core fix.

---

## 0. TL;DR — the one root cause

Cross-topic contamination is **not** a generation bug you can prompt away, and it is **not** a bug in any single filter. It is a **layering bug**: off-topic content enters the model's prompt at *retrieval/context-assembly time*, and every defense you have built fights it *after generation* with hardcoded string lists that are structurally blind to the most common case.

Three independent code facts create the failure, and they compound specifically in point-based answers:

1. **The retrieval filter has a permanent hole.** `multi_source_rag.py:4417` filters chunks with `{"policy_type": {"$in": [query_type, "general"]}}`. Every `"general"`-tagged chunk passes for **every** query. Per your own KB re-classification work, ~40% of chunks are `"general"` — because a single-topic tag *cannot* represent a chunk that genuinely discusses several types (an underwriting-practices chunk that uses a marine example, a glossary, a "Lesson Round Up" chapter summary that lists fidelity insurance, etc.).

2. **Generation is fed the FULL chunk, not the relevant part.** `multi_source_rag.py:5382` builds the prompt from `chunk.page_content` — the entire chunk. So when a mixed `"general"` chunk is retrieved for a motor query, its embedded *marine* sentence is sitting right there in the prompt. Notably, `_build_grounding_context` (`:2362`) already extracts only the **query-relevant window** of each chunk via `_rerank_windows` for the grounding check — but generation ignores that and uses the whole chunk.

3. **The `DETAILED_GROUNDED_PROMPT` turns every context fragment into a list item.** `prompt_template.py:284-293` instructs a numbered list covering "the points the KNOWLEDGE BASE actually makes." A 7B model (Qwen2.5-7B-Instruct-AWQ) reads a mixed chunk and dutifully emits the off-topic sentence as **point N**. Rule 2 ("use generic material only to support a point") is a soft instruction the small model silently ignores — a pattern you have already documented repeatedly.

**Why point-based answers specifically:** in prose/brief mode the model *synthesizes* and naturally drops weak/off-topic fragments. A numbered list does the opposite — it enumerates one point per distinct fact in context, so any off-topic fragment becomes an explicit standalone claim. Detailed mode also widens the retrieval net (`_doc_top_k` 18 vs 12, `_chunk_limit` 8 vs 5 at `:4342`/`:4358`), pulling in *more* chunks and therefore more contamination surface. Format + wider net = contamination is visible almost only here.

**Why your defenses don't catch it (the whack-a-mole explained):** the post-generation contamination filter (`_text_has_giveaway_contamination`, `:1304`) is gated on `_retrieved_specific_types` (`:5357`), which **explicitly excludes `"general"`**. So when contamination comes from a `"general"` chunk — the dominant case — the actual contaminating type (`"marine"`) is not in the set, and the filter skips it (`:1341` `if _type_name not in retrieved_types: continue`). The filter only ever fires by *coincidence*: when a separately-tagged chunk of the contaminating type also happened to be retrieved. On top of that it is a hardcoded 6-type dictionary (`_TYPE_GIVEAWAY_TERMS`, `:1282` — marine/health/crop/fidelity/transit/motor) with hand-listed jargon; any of the ~10+ other KB types (aviation, engineering, cyber, liability, fire, personal accident, travel, life, burglary…) contaminating an answer is caught by **nothing**. Every "new query breaks it" episode is a new type or a new phrasing the list never covered.

> **The fix is to stop contamination from entering the prompt (levers 1+2), and replace the entire hardcoded-list post-filter with one dynamic, type-agnostic semantic check (lever 3). No new jargon list. No per-type patching.**

---

## 1. Evidence map (read these before building)

| # | Fact | Location |
|---|------|----------|
| A | Hard filter always admits `"general"` | `multi_source_rag.py:4413-4418` |
| B | Soft down-weight is only `0.5×` — loses ties but a strong mixed chunk survives | `multi_source_rag.py:4675`, `_effective_sort_score` `:4722-4726` |
| C | Generation prompt uses **full** `chunk.page_content` | `multi_source_rag.py:5382` |
| D | Grounding check already uses **query-relevant windows** (reusable mechanism) | `_build_grounding_context :2362`, `_rerank_windows` in `turbovec_store.py:202` |
| E | Detailed prompt maps chunks → numbered points; soft "use generic only to support" rule | `prompt_template.py:284-293` (rule 2 at `:270`) |
| F | Post-gen contamination filter is hardcoded 6-type jargon dict | `_TYPE_GIVEAWAY_TERMS :1282`, `_text_has_giveaway_contamination :1304` |
| G | Filter is gated on retrieved types that **exclude** `"general"` → blind to the common case | `_retrieved_specific_types :5357-5360`, gate at `:1341` |
| H | Existing per-point filter checks grounding (point↔context overlap), **not** topic relevance (point↔query) | `_point_grounded :6120` |
| I | Shared cross-encoder reranker available in-process for cheap per-point scoring | `_get_shared_reranker()` / `CrossEncoder.predict` — `turbovec_store.py:104`, `:990` |
| J | Detailed mode widens retrieval → more contamination surface | `:4342`, `:4358` |

Stacked post-gen filters that all share the same blind spot and will be consolidated: ungrounded-point (`:6120`), cross-topic detailed (`:6179`), history-contamination brief (`:6268`), retrieval-contamination brief (`:6319`), third-party-victim (`:7261`).

---

## 2. Design principles (the "dynamic, doesn't break" contract)

1. **Prevent, don't scrub.** Stop off-topic text from reaching the prompt. A point that is never generated can never leak.
2. **One semantic signal, zero hardcoded lists.** Topic relevance is measured by the cross-encoder already in the pipeline (query ↔ text), which is type-agnostic and works for insurance types that don't exist in the KB yet. No `_TYPE_GIVEAWAY_TERMS`, no per-type exemptions to maintain.
3. **Measure before and after, on a class not an instance.** Every change is validated against a labeled contamination corpus and a single contamination-rate metric — never against one query. This is what breaks the whack-a-mole loop.
4. **Fail toward recall, tune deliberately.** Thresholds are calibrated from measured score distributions (like the existing `0.5`/`0.05` rescue bars were), not guessed. A relevance gate must not silently drop legitimately-relevant-but-differently-worded content (the `liability` recall lesson at `:4312-4341`).
5. **Consolidate only after the new gate proves out.** Keep the old filters running until the semantic gate demonstrably subsumes them on the corpus, then delete them in one pass. No big-bang removal.

---

## 3. The plan — phased

### Phase 0 — Observability harness (build FIRST; this is what has been missing)

You cannot fix blind, and every prior fix was blind. Before touching behavior, build the instrument that makes the failure and any fix **visible and measurable**.

**0a. Per-answer trace.** Add an opt-in debug trace (env flag, e.g. `CONTAMINATION_TRACE=1`, mirroring the existing `TIMING` log pattern) to `ask_stream` that, for each detailed/point answer, logs a structured record:
- the query, `_query_policy_type`, and `_query_candidate_type`;
- every retrieved chunk: source, `policy_type`, `candidate_policy_type`, `rerank_score`, whether it survived to context;
- every generated point mapped to its **best-matching source chunk** (via the same word-overlap / window logic already in `_point_grounded`) and that chunk's `policy_type`;
- a per-point **query-relevance score** from the cross-encoder (see Phase 2) — logged even before it gates anything.

Output as one JSON line per answer to a rotating file under the data dir. This immediately answers the question you actually have — *"is it working according to plan?"* — by showing, per contaminated point, which chunk it came from and what that chunk was tagged.

**0b. Labeled contamination corpus.** Assemble ~40-60 queries that historically triggered contamination (mine your memory files + the confirmed-live cases already quoted in code comments: "Explain motor insurance in detail" → marine/health leak `:6153-6162`; "Explain engineering insurance in detail" → hull/marine `:6163-6166`; "Explain burglary insurance in detail" → fidelity `:6182-6199`; personal-accident/motor; term-insurance/driving; etc.). Store as a JSON fixture: `{query, expected_topic, forbidden_topics/phrases}`.

**0c. Contamination-rate metric + runner.** A script that runs the corpus against the live backend, parses each answer's points, and scores contamination using the **new semantic signal** (not the old jargon list) plus the forbidden-phrase labels, emitting a single number: `% of answers containing ≥1 off-topic point`. This is the north-star metric every later phase moves. Baseline it now.

**Exit criteria for Phase 0:** trace produces readable per-point attribution on a live contaminated repro; corpus + metric run green with a recorded baseline contamination rate.

---

### Phase 1 — Window-scoped generation context (upstream lever, highest leverage, lowest risk)

Stop feeding the model the off-topic portions of mixed chunks.

- At context assembly (`:5364-5382`), replace the full-`page_content` insertion with the **query-relevant window(s)** of each chunk, reusing `_rerank_windows` (`turbovec_store.py:202`) exactly as `_build_grounding_context` (`:2362`) already does. Use a generous window (e.g. top-1 window, or top-2 concatenated) so genuine detail isn't lost.
- Keep full-chunk text for chunks whose window selection is degenerate (very short chunk, single window) — windowing a 2-sentence chunk buys nothing and risks truncation.
- This directly removes the "marine sentence embedded in a general underwriting chunk" from the prompt, so it can never become a point — without excluding the chunk (its on-topic window still contributes).

**Risk:** windowing could drop a needed second fact from a legitimately on-topic chunk. Mitigation: tune window count on the Phase-0 corpus **and** a recall spot-check set (the liability/engineering "differently-worded chunk" cases). Measure both contamination rate ↓ and answer-completeness (no regression on known-good detailed answers).

**Exit criteria:** contamination rate drops materially vs. Phase-0 baseline with no measurable loss on the recall spot-check set.

---

### Phase 2 — Semantic per-point topic-relevance gate (replaces the hardcoded stack)

This is the dynamic, type-agnostic core. It catches the residual contamination that survives Phase 1 (a mixed chunk whose *relevant* window still contains a comparative off-topic clause, or a point the model half-invents from cross-topic priming).

**Mechanism:**
- After the answer is generated and split into points (reuse the exact opener/points/closer split already at `:6207-6232` and `:6034-6046` — do not write a third splitter), score **each point against the query** with the shared cross-encoder: one batched `CrossEncoder.predict([(query, point_1), (query, point_2), …])` call via `_get_shared_reranker()`. One `.predict()` for ≤8 points is cheap because the model is already resident in-process.
- Drop a point when its query-relevance score is an outlier *below* the answer's own cohort — i.e. relative to the median/max point score for this answer, not an absolute global constant — with an absolute floor as a backstop. A genuinely on-topic answer has all points clustered high; a contaminated point sits visibly below its siblings. Relative scoring is what makes this robust to the fact that absolute rerank scores vary a lot by topic (the same reason the codebase already prefers relative reasoning at `:4728-4762`).
- Never drop the last surviving point (avoid emptying an answer); if all points score low, that's a retrieval/grounding failure and should fall through to the existing refusal path, not produce an empty list.

**Why this is strictly better than `_TYPE_GIVEAWAY_TERMS`:**
- Works for **any** insurance type, including ones not in the KB yet — no list to maintain, no per-query patch.
- Catches contamination from `"general"` chunks (the case the current filter is structurally blind to) because it never consults chunk tags at all — it asks "is this sentence about what the user asked?"
- Symmetric with how retrieval already ranks relevance, so its behavior is predictable and calibratable from the same score distributions.

**Risk:** false positives (dropping a legitimately-relevant point that happens to score low), and latency of one extra `.predict`. Mitigations: calibrate the relative-drop threshold on the Phase-0 corpus + a "clean detailed answers" control set (must drop 0 points on those); gate the whole check to detailed/point answers only; log every drop via the Phase-0 trace so regressions are visible.

**Exit criteria:** on the corpus, contamination rate approaches ~0; on the clean-control set, **zero** legitimate points dropped; added latency within budget (measure against the existing `TIMING` line).

**OUTCOME (2026-07-22 through 2026-07-24) — exit criteria NOT met, phase closed:**
First attempt (delete-based, ungated) actively deleted a real point from a
correct answer ("Explain life insurance in detail" lost "Assignment
differs from Nomination") on a clean-control retest — reverted same day.
Redesigned as a guarded, demote-not-delete gate (confidence floor +
foreign-type confirmation before acting, reorders to the end instead of
removing). Confirmed safe across two full-corpus sweeps (zero real
content loss even on its one false-positive firing), but never shown to
work: a 112-run sweep read 0% contamination with the gate active, but a
follow-up 280-run sweep (56 cases x 5 repeats,
`contamination_guarded_gate_phase2_bigsweep.json`) showed that "0%" was
sample-size noise, not a real effect — rates with the gate active
(overall 0.71%, repro 3.33%) were statistically indistinguishable from
the gate-off baseline (0.89%, 4.17%). The gate itself fired only twice in
280 runs: one confirmed false positive on a clean control, one ambiguous
demotion on a repro case uncreditable from n=1. **Verdict: real cost
(added complexity, a nonzero false-positive rate) for zero demonstrated
benefit.** Reverted to `POINT_RELEVANCE_GATE_ACTIVE=0` in `.env`, with
the full history and this verdict documented there so it isn't
re-enabled on a small sample again. If this is revisited, it needs a
genuinely different detection approach, not a threshold retune of the
same per-point cross-encoder score — that axis was tested at real scale
and didn't separate contaminated points from clean ones distinctly
enough to act on.

---

### Phase 3 — Consolidate and retire the brittle filters

**SKIPPED (2026-07-24)** — this phase's own stated precondition
("only after Phase 2 proves out") was not met; see the Phase 2 outcome
above. Deleting `_TYPE_GIVEAWAY_TERMS` / `_text_has_giveaway_contamination`
/ etc. now, with nothing proven to replace them, would remove real
working defenses for no gain and very likely make contamination worse,
not better — the opposite of this phase's own exit criteria. Left as
future work, gated on a future Phase 2 redesign actually proving out,
not on a deadline. Original phase description kept below for reference.

Only after Phase 2 proves out on both corpus and control set:
- Delete `_TYPE_GIVEAWAY_TERMS` (`:1282`), `_TYPE_QUERY_EXEMPT_WORDS` (`:1299`), `_text_has_giveaway_contamination` (`:1304`), and the detailed/brief/history/retrieval contamination blocks that depend on them (`:6179`, `:6268`, `:6319`) — replaced by the single Phase-2 gate.
- Evaluate the third-party-victim block (`:7261`) separately: it encodes a *narrative-structure* leak (first-party example describing a third-party victim), which the point-relevance gate may not fully cover. Keep it until the corpus shows the semantic gate catches its cases; otherwise fold its intent into the Phase-2 scoring (score the point against the query, which a "pedestrian gets paid" point for a "personal accident cover" query should already fail).
- Keep `_point_grounded` (`:6120`) — grounding (point↔context) and topic-relevance (point↔query) are **different axes**; you want both. Just make sure they run in a defined order (ground first, then topic-gate) and share the split.

**Exit criteria:** net **reduction** in filter code; corpus + control set still green; no regression across a broad manual sweep.

---

### Phase 4 — Lock it against future breakage (the "doesn't break" guarantee)

- Wire the Phase-0 corpus + contamination metric into a **repeatable regression command** (script under `RAG_InsureAI/`, same spirit as the existing deploy/verify cycle). Running it is the definition of done for any future retrieval/prompt change.
- Document the single knob (the relative-drop threshold + absolute floor) and how it was calibrated, so it is tuned from data, never guessed.
- Add a short note to `CLAUDE.md`/memory: contamination is now a *measured rate on a corpus*, not a per-query bug — fixes must move the metric, and new contamination reports get **added to the corpus first**, then fixed. This institutionalizes the escape from whack-a-mole.

**DONE (2026-07-24):** `contamination_corpus_runner.py` is now the regression
command — see its own module docstring for usage and the calibrated
PASS/FAIL criteria (hard fail on any control-set contamination, warn above
2x the known-good repro baseline). It exits non-zero on a hard regression,
so it can gate a deploy rather than needing a human to read percentages
every time. "The single knob" from the original bullet doesn't apply as
written — Phase 2 (the thing that knob belonged to) didn't pan out and is
off — so there's no active Phase-2 threshold to document; the real,
currently-active defenses are Phase 1's window-scoping and the existing
hardcoded filters (`_TYPE_GIVEAWAY_TERMS`, `_EXCLUSION_LANGUAGE_RE`), which
have no comparable single knob to calibrate. No repo-level `CLAUDE.md`
existed to add the "measured rate, not a per-query bug" note to, so it's
recorded here instead, plus in this session's memory: **the definition of
done for any future contamination-adjacent change in this codebase is a
`contamination_corpus_runner.py --repeats 5` PASS, not "the one query I
tried looks right."**

---

## 4. Files touched

| File | Change |
|------|--------|
| `multi_source_rag.py` | Phase 1 window-scoped context at `:5364-5382`; Phase 2 per-point cross-encoder gate near the existing point filters `:6034-6266`; Phase 0 trace hooks; Phase 3 deletions |
| `prompt_template.py` | Optional: tighten `DETAILED_GROUNDED_PROMPT` rule 2/7 wording, but treat prompt as *non-load-bearing* — the 7B model ignores soft rules, so this is cosmetic, not the fix |
| `turbovec_store.py` | None expected — reuse `_get_shared_reranker` / `_rerank_windows` as-is |
| new: `app/contamination_trace.py` (or inline) | Phase 0 structured trace writer |
| new: `RAG_InsureAI/contamination_corpus.json` + runner script | Phase 0/4 metric harness |

---

## 5. Key decision points (flag to user before/while building)

1. **Window count in Phase 1** — top-1 (aggressive, max contamination reduction, small recall risk) vs top-2 (safer recall, slightly more surface). Recommend starting top-2, tightening to top-1 only if the corpus shows residual chunk-internal leakage. *Decide from measured data, not upfront.*
2. **Phase 2 threshold policy** — pure relative (drop points > X below the answer's median) vs relative-with-absolute-floor. Recommend relative-with-floor. The exact numbers come out of calibration in Phase 2, not this document.
3. **How aggressively to retire old filters in Phase 3** — recommend keeping `_point_grounded` and the third-party-victim block, deleting the rest. Confirm before mass deletion.
4. **Latency budget** — one extra `.predict` per detailed answer. Confirm acceptable against current `TIMING` numbers (expected: negligible, model resident, ≤8 short pairs).

---

## 6. Risks & honest tradeoffs

- **Recall regression from windowing (Phase 1)** is the real risk. The codebase has a documented history of narrowing retrieval and losing correct-but-differently-worded chunks (`:4312-4341`). Windowing is milder (it trims *within* a kept chunk, doesn't drop chunks) but must be measured against a recall control set, not just the contamination corpus.
- **False-positive point drops (Phase 2).** A legitimately terse or oddly-phrased on-topic point could score low. The relative-cohort scoring + never-drop-last-point rule + zero-drop requirement on the clean-control set are the guardrails. The Phase-0 trace makes every drop auditable.
- **This does not fix "answer generation is very bad" in general** — it fixes cross-topic contamination in point answers, the stated core issue. If broad generation quality (tone, padding, vagueness) is also in scope, that is a separate workstream; flag it and scope separately rather than bundling.
- **Semantic gate depends on the cross-encoder's judgment.** It is far more robust than a hardcoded list, but not infallible; that's exactly why Phase 0's metric and Phase 4's regression harness exist — so its failures are visible and correctable by threshold, not by adding another list.

---

## 7. Orchestration (after this plan is approved)

- Sonnet implements phase by phase, in order. Each phase has explicit exit criteria above; do not advance until met.
- After each phase, I (Opus) verify against the Phase-0 corpus/metric and a live browser repro of the canonical cases ("Explain motor insurance in detail", "Explain engineering insurance in detail", "Explain burglary insurance in detail"), deploy via the existing `docker cp` + `docker restart insurehub_api` cycle, and purge `query_kv_cache.json` before retesting (established gotcha — cached answers mask fixes).
- Phase 0 lands and baselines *before* any behavior change, so every later number is comparable.
- Definition of done: contamination rate on the corpus at/near zero, zero legitimate-point drops on the control set, old filter code net-reduced, regression command wired and documented.
