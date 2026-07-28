# Opus review — latency work vs. plan_latency.md

**Reviewer:** Opus 4.8 · **Implementer:** Sonnet · **Commits reviewed:** `667d5b8`, `018adb1`, `90e657f`, `6f1d66e`

**Verdict: not yet done.** The code is directionally right and matches the plan's design, but **none of the latency work has ever actually executed**, and the measurement harness has defects that would produce a silently-empty or wrong baseline. Fix the blockers before trusting any number.

---

## What is correct (verified, no action needed)

- **Device auto-detect logic** (`turbovec_store.py`) — both named load sites get `device=`, `MODEL_DEVICE` override works, defaults to auto. Verified live: both models load *and run real inference* with `device="cpu"` on this host. Correctly a no-op on CPU.
- **Compressor inherits the shared embedder** — confirmed both `ContextCompressor(...)` sites (`multi_source_rag.py:3496`, `rag.py:604`) pass `vector_store.embed_model`, i.e. the shared getter's instance. The plan's fact-E claim ("compressor gets GPU for free, no separate change") **holds**.
- **TTFT placement is semantically right** — set once on the first *content* token, and the buffered/non-streaming fallback path is covered.
- **Runner conventions** — mirrors `contamination_corpus_runner.py`, checks `DISABLE_QUERY_CACHE` and warns loudly. Good.
- **Contamination Phase 2 isolation redesign is live and behaving as designed** — `drop=1/8 (gap=5.24x)`, `drop=2/8 (gap=14.83x)`: genuine minorities, versus the old ratio-to-max design's 6/8 false positive.

---

## BLOCKERS — fix before any baseline is trusted

### B1. The TTFT + GPU code has never actually run
The container started `2026-07-22T12:10:14Z` (17:40 IST); commit `90e657f` landed 17:54 IST. Live TIMING lines carry **no `ttft=` field**. The code is on disk inside the container (bind mount) but was never loaded by the running process — it is **compile-checked only, never observed working**. This violates the plan's own "verify live" discipline, and the entire Phase 0 premise depends on `ttft=` existing.
**Fix:** restart `insurehub_api`, then confirm a real `ttft=` value appears in a TIMING line *and* that `[SharedModels] ... (device=cpu)` logs on model load.

### B2. The runner requires `ttft=` and will match zero lines against un-reloaded code
`_TIMING_RE` has `ttft=(\S+)` as a **mandatory** group. Against the currently-running process every case returns `{}` → the table prints all `?` with no error. Running the harness right now yields a **silently empty baseline that looks like a successful run**.
**Fix:** make the harness fail loudly when it matches zero TIMING lines (or make `ttft=` optional and report it as missing), so an un-deployed build can never masquerade as a result.

---

## DEFECTS — would produce wrong or missing data

### D1. The follow-up case can never match its TIMING line
The runner matches by `query[:40] in line`, but the TIMING line logs **`retrieval_query`**, which is *rewritten* for follow-ups (`multi_source_rag.py:4353-4379` merge/reformulate; `_correct_typos` at `:4464` runs on every query). For the follow-up case the measured text (`"What's excluded under it?"`) never appears in the log line — the rewritten standalone query does.
**Fix:** log the original question (or a per-request id) alongside `retrieval_query`, and match on that. The runner already generates a unique `session_id` — logging and matching on it is the cleanest fix, and also solves D4.

### D2. The refusal case can never produce a TIMING line at all
Gated refusals `yield ... ; return` **before** reaching the TIMING log (e.g. the reranker-gate refusal). Confirmed empirically earlier this session: a refusal query emitted no TIMING line. The plan asked for a refusal case in the 6-query set, but the instrumentation deliberately skips early returns (documented at `:3974`). As written this row is **permanently blank**.
**Fix:** either instrument the early-return paths (at minimum total + ttft), or drop the refusal case from the set and document why. Do not leave a case that silently never reports.

### D3. Follow-up case reuses one `session_id` across cold and warm passes
The warm pass therefore runs as turn 3–4 with the cold pass's history already in the session, not as a clean 2-turn conversation. Cold and warm are measuring **different conversational states**.
**Fix:** fresh `session_id` per pass.

### D4. Log-window fragility
`docker logs --tail 60` may not contain the target TIMING line on a busy container — each request emits several INFO lines (Compressor, agent_hub, TIMING).
**Fix:** widen the window, and match on the unique `session_id` (see D1) rather than query text.

---

## GAPS vs. the plan (incomplete, not wrong)

### G1. Phase 1a is code-only — the infra half is missing
The plan's Phase 1a explicitly requires the image to ship **CUDA-enabled torch** and the runtime to **expose the GPU** (`--gpus all` / compose `deploy.resources.reservations.devices`). Neither `Dockerfile` nor `docker-compose.yml` was touched. Without that, `torch.cuda.is_available()` stays `False` on the GPU server and the whole change is a **silent no-op there** — it will look deployed and do nothing.
**Fix:** land the image/compose GPU config. Phase 1a is not done until this exists and `device=cuda` is observed in the server's logs.

### G2. Third model-load site missed
`semantic_chunker.py:56` builds `SentenceTransformer(EMBED_MODEL_NAME)` with no `device=`, bypassing both the shared getter and the device fix. It is a lazy fallback (fires only when no `embed_model` is passed) and sits on the **ingestion** path, not the answer-latency path — so it does not affect the 5–6s goal — but it stays CPU-bound on a GPU host and holds a duplicate model copy in memory.
**Fix:** pass the shared model / device. Low priority, but the plan's Phase 1a said "the two shared-model load sites" — that enumeration was itself incomplete.

---

## MINOR

- **M1.** `_ms()` returns `None` for `"n/a"`; the table does `w.get('total_ms', '?')`, so a present-but-`None` value prints `None` rather than `?`. Cosmetic.
- **M2.** `_purge_kv_cache()` is redundant when `DISABLE_QUERY_CACHE=1`. Harmless belt-and-braces.

---

## PROCESS

- **P1.** Commit `90e657f` bundles **two unrelated plans** — latency TTFT instrumentation *and* the contamination Phase 2 statistic redesign. The plan's own principle is "one change at a time, re-measured"; bundling makes attribution and rollback harder. Not worth rewriting history; worth not repeating.
- **P2.** The Phase 2 corpus sweep **never completed** (stopped, no `phase2_isolation_gate_sweep.json` written). The isolation-gate redesign's broad validation is still outstanding, even though live logs show it behaving sensibly. Re-run before considering contamination Phase 2 settled.

---

## Recommended order for Sonnet

1. **B1** — restart, confirm `ttft=` and `device=` actually appear. Nothing else is measurable until this is true.
2. **B2, D1, D2, D3, D4** — fix the harness (session-id matching + loud failure) so a baseline can't be silently empty or wrong.
3. Run the **real Phase 0 baseline** on an idle backend, commit the table.
4. **G1** — GPU image/compose config, then re-measure on the server (the only place the GPU win is visible).
5. **G2, M1, M2** — cleanup.
6. **P2** — re-run the contamination Phase 2 sweep.

---
---

# Round 2 — Opus review of `9e97aea` (Sonnet's fixes) + the first baseline

**Verdict: the harness bugs are genuinely fixed; the baseline they produced is not yet trustworthy.**
B1/B2/D1/D2/D3/D4/M1 all verified closed. But the committed baseline has one
instrumentation line that reports a factually wrong number, and three
methodological problems that make it unsafe to use as the before/after
reference Phase 4 depends on.

## Confirmed fixed (checked, not taken on trust)

- **D1** — `question=%r` present at all three TIMING sites. The regex change to
  `query=(?:'[^']*'|"[^"]*")` is non-capturing, so groups 1–10 keep their
  meaning; the apostrophe case (`question="What's excluded under it?"`, where
  `%r` flips to double quotes) matches. Live-confirmed on both a generation and
  a refusal request.
- **D2** — reranker-gate refusal instrumented; Sonnet then found live that the
  plan's own refusal query actually exits at the *grounding-check* refusal
  (`:5363`), not the reranker gate, and instrumented that too. Catching that
  required running it, not reading it. Good.
- **D3** — fresh `session_id` per `(case, pass)`; confirmed in the logs
  (`latency-baseline-refusal-warm-e17decf0`).
- **B2** — `TimingNotFound` + `sys.exit(1)`. This fix *proved itself*: it is what
  surfaced the stderr bug below instead of printing a plausible-looking table.
- **D4** — tail 60 → 300. **M1** — `_cell()` helper.
- **The stderr bug** — `docker logs` with `capture_output=True` split stdout from
  stderr, and 100% of this app's log output (every TIMING line) is on stderr.
  A pure code review could not have found this; only running it could.
- **Phase 0's literal exit criterion** ("committed clean per-phase + TTFT baseline
  table") is met.

## R1 — BLOCKER (correctness): the refusal TIMING line reports `other=0ms` while ~48% of its wall clock is unaccounted

Every generation row balances to within 1–3 ms. Both refusal rows do not:

| case | total | retrieval | grounding | preprocess | accounted | **gap** | reported `other` |
|---|---|---|---|---|---|---|---|
| refusal cold | 13347 | 4908 | 983 | 759 | 6650 | **6697** | `0ms` |
| refusal warm | 11920 | 4938 | 551 | 637 | 6126 | **5794** | `0ms` |

Cause is structural, not a rounding artifact. `_t_promptbuild_start` is set at
`:5086` (right after the grounding gather) and `_t_promptbuild_ms` is assigned
**only** at `:5874` — far past both refusal exits (`:5014`, `:5377`). Everything
between is unmeasured, and on a refusal that span is the entire pre-refusal
retry cascade:

`_vllm_clean_query` (LLM call) → optional fallback `_retrieve_doc_chunks` +
rerank → `_verify_grounding_any_chunk` (LLM call) → standalone-retry
`_retrieve_doc_chunks` → `_verify_grounding_any_chunk` (LLM call)

Both new lines then pass a hardcoded literal `0` for `other=%dms`, so the log
actively asserts that gap is zero. The main TIMING line at `:7907` computes
`other = total − (retrieval + grounding + llm)` and its own comment explains
that the retry tiers land in `other` **by design** — the new lines broke that
contract.

**Consequence:** the committed table says refusals are retrieval-dominated
(4.9 s of 11.9 s). In fact the single largest cost is a ~5.8 s chain of extra
LLM round-trips that does not appear anywhere. Optimising from this table
optimises the wrong phase.

**Fix:** compute `other` the same way `:7907` does. Recommended follow-up: give
the retry cascade its own field — it is ~48% of refusal latency and, when a
retry fires on a *generation* path, it is currently silently folded into
`promptbuild`, which mislabels it there too.

## R2 — the baseline is n=1 per cell against a remote, shared vLLM host

`VLLM_HOST=http://123.253.124.14:7000` is remote; "idle backend" cannot be
enforced for it. Two full runs today, **zero code change between them**:

| case | run 1 warm | run 2 warm | spread |
|---|---|---|---|
| brief_1 | 55307 | 15242 | **3.6×** |
| followup | 17187 | 25810 | 1.5× |
| detailed_2 (cold) | 39802 | 55853 | 1.4× |
| detailed_1 | 37035 | 36362 | 1.02× |

Generation-heavy cases are fairly stable (±5–40%), but `brief_1` swung 3.6×.
Phase 4 wants to re-run this and attribute the delta to a code change. A single
sample can be a 3.6× outlier, so it cannot support that.
**Fix:** N ≥ 3 measured reps per cell; report median plus spread.

## R3 — the cold/warm split measures nothing and doubles runtime

`_purge_kv_cache()` clears `query_kv_cache.json` only. The thing "warm" exists to
warm — the compressor's in-process `_sent_cache` — is never reset between
passes, cases, or runs. Logs show it accumulating monotonically
(`cache size=8/3000` → `15/3000` …) and surviving from earlier manual testing,
so the "cold" pass is already warm. Retrieval cold → warm across the six cases:
−8%, −15%, +8%, −4%, −6%, +1% (mean ≈ −4%, i.e. noise).
**Fix:** drop the split and spend the doubled runtime on R2's repeats, keeping a
single discarded warm-up per query (which is what the plan's "run twice, take
the 2nd" actually asks for).

## R4 — the harness cannot evaluate Phase 3 at all

`run_case` calls `_ask(turn, session_id)` and discards the return value, so no
answer text, length, or finish reason is recorded. Phase 3's exit criterion is
*"detailed answers tighter/complete, `total` unchanged-or-better, no quality
regression"* — none of which is measurable from what is captured.

Worse, `llm=` is confounded by output length: `VLLM_MAX_TOKENS=512` at the
measured 7–8 tok/s is ~64 s for a full cap, and `brief_1`'s 49 s outlier is
consistent with hitting it. Without token counts, "the system got slower" and
"the model wrote more" are indistinguishable.
**Fix:** record answer length + finish reason per run.

## R5 — match is against the raw line, not the captured question

`_read_timing_for_question` filters with `needle not in line`, testing the whole
line — including the `query=` field — while the regex already captures the
question as **group 11 and never uses it**. `brief_1`'s text is byte-identical to
`followup`'s turn 1, so the set already contains duplicate question text; it is
one reordering away from a silent cross-match. No live collision in the current
run (verified).
**Fix:** validate against `m.group(11)`.

## R6 — 2 of 11 refusal exits are instrumented

Eleven sites yield the refusal string; two now emit TIMING. With B2 now hard-
failing on zero matches, a query that drifts to any other refusal exit fails the
**whole** run rather than one row. That is the right trade over silence, but it
makes the fixed 6-query set brittle to KB changes.

## R7 / R8 — minor

- **R7.** Both refusal lines call `time.time()` twice for what is meant to be one
  instant (`_t_ttft_ms`, `_t_total_ms`). Use one.
- **R8.** `ttft` means different things per row: a strict prefix of `total` on
  generation rows, exactly equal to `total` on refusal rows. Unmarked in the
  table; averaging the column would be meaningless.

## What the baseline *does* establish reliably

The ratios are stable across both runs even where absolute numbers are not, and
they confirm the plan's sequencing:

- generation = **61–77%** of total on detailed queries (llm 26.6–33.5 s of 36–43 s) → plan Decision 3 (GPU vLLM) is the dominant lever, as written;
- retrieval = **5–8 s**, the clear second → Phase 1a (GPU embedder/reranker);
- grounding < 1 s, promptbuild < 0.4 s → not worth touching.

So the strategy is safe. The individual numbers are not yet a baseline.

## CORRECTION to Round 1 — G1 was wrong

Round 1 stated *"Neither `Dockerfile` nor `docker-compose.yml` was touched."*
**That was false.** Commit `0e37e8f` (2026-07-21, a day *before* the review)
already added:

- `Dockerfile` — `ARG TORCH_DEVICE=cpu` with a real CUDA branch installing torch
  from the `cu121` index, placed *before* `requirements.txt` so pip cannot
  resolve a conflicting CPU wheel over it;
- `docker-compose.yml` — `TORCH_DEVICE: ${TORCH_DEVICE:-cpu}` build arg;
- `docker-compose.gpu.yml` — a complete overlay setting `TORCH_DEVICE: cuda`
  plus `deploy.resources.reservations.devices` for both services, deliberately
  leaving the base compose CPU-safe.

I did not check for a GPU overlay file before asserting its absence. The
CUDA-wheel and device-reservation halves of G1 were done.

**The real gap was one level down: nothing in the deploy path used the overlay.**
`deploy-server.sh` ran `docker compose build api` / `up -d api` with no `-f`
flags, and wrote a systemd unit doing the same — so on a GPU host the documented
deploy still built the CPU image, and any reboot would have reverted a manual
GPU deploy back to CPU. Same silent-no-op failure Round 1 described, located in
the deploy script rather than the compose files.

**Fixed:** `deploy-server.sh` now probes for a GPU (host `nvidia-smi` *and* a real
`docker run --gpus all` check, since the first does not imply the second),
selects the overlay accordingly with `FORCE_GPU` / `FORCE_CPU` overrides, carries
the same file list into the systemd unit, and asserts the plan's Phase 1a exit
criterion after deploy by grepping the container's own `device=` log line —
reporting loudly if a GPU build ended up on CPU.

## Still open

- **Decision 3** — relocating generation onto a GPU-hosted vLLM. Not a compose
  change: `VLLM_HOST` points at an external `:7000` that this repo does not
  manage. Still the single largest lever (61–77% of detailed-query latency).
- **Phase 1a verification** — the GPU path cannot be exercised from this Mac.
  Unverified until it runs on the server.
- **R6** — 9 uninstrumented refusal exits.
- **P2** — contamination Phase 2 corpus sweep re-run.
